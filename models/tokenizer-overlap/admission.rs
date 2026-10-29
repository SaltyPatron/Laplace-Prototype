use super::{Error, Id, Views, hex};
use serde_json::json;
use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char, c_int, c_void},
    fs, ptr,
    time::Instant,
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Reference {
    id: Id,
    coordinate: [i64; 4],
    tier: u8,
    said: u8,
    outcome: u8,
    position: u32,
    spare: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Vertex {
    id: [u8; 16],
    run: u32,
    said: u32,
    spare: u32,
    outcome: u32,
    position: u32,
}
impl Vertex {
    fn id(&self) -> Id {
        Id([
            u64::from_ne_bytes(self.id[..8].try_into().unwrap()),
            u64::from_ne_bytes(self.id[8..].try_into().unwrap()),
        ])
    }
}

#[link(name = "laplace_core")]
unsafe extern "C" {
    fn canonical_stage_new(
        nodes: usize,
        vertices: usize,
        bytes: usize,
        highway: *const c_void,
    ) -> *mut c_void;
    fn canonical_stage_free(stage: *mut c_void);
    fn canonical_stage_reference(stage: *mut c_void, value: *const Reference) -> i32;
    fn canonical_stage_text(
        stage: *mut c_void,
        text: *const u8,
        length: usize,
        out: *mut Reference,
    ) -> i32;
    fn canonical_stage_compose(
        stage: *mut c_void,
        children: *const Reference,
        count: usize,
        tier: u8,
        entity_mask: u64,
        path_mask: u64,
        spine: i32,
        out: *mut Reference,
    ) -> i32;
    fn canonical_stage_begin(stage: *mut c_void, roots: *const Id, count: usize) -> i32;
    fn canonical_stage_frontier(stage: *const c_void, count: *mut usize) -> *const Id;
    fn canonical_stage_answer(stage: *mut c_void, present: *const u8, count: usize) -> i32;
    fn canonical_stage_finish(stage: *mut c_void) -> i32;
    fn canonical_stage_copy(
        stage: *const c_void,
        table: i32,
        bytes: *mut usize,
        rows: *mut usize,
    ) -> *const u8;
    fn canonical_stage_columns(table: i32) -> *const c_char;
    fn lp_path_vertices(
        bytes: *const u8,
        length: usize,
        out: *mut Vertex,
        capacity: usize,
    ) -> usize;
}
#[link(name = "pq")]
unsafe extern "C" {
    fn PQconnectdb(conninfo: *const c_char) -> *mut c_void;
    fn PQstatus(connection: *const c_void) -> c_int;
    fn PQfinish(connection: *mut c_void);
    fn PQerrorMessage(connection: *const c_void) -> *const c_char;
    fn PQexecParams(
        connection: *mut c_void,
        sql: *const c_char,
        count: c_int,
        types: *const u32,
        values: *const *const c_char,
        lengths: *const c_int,
        formats: *const c_int,
        result_format: c_int,
    ) -> *mut c_void;
    fn PQresultStatus(result: *const c_void) -> c_int;
    fn PQresultErrorMessage(result: *const c_void) -> *const c_char;
    fn PQntuples(result: *const c_void) -> c_int;
    fn PQgetvalue(result: *const c_void, row: c_int, column: c_int) -> *const c_char;
    fn PQclear(result: *mut c_void);
    fn PQputCopyData(connection: *mut c_void, bytes: *const c_char, length: c_int) -> c_int;
    fn PQputCopyEnd(connection: *mut c_void, error: *const c_char) -> c_int;
    fn PQgetResult(connection: *mut c_void) -> *mut c_void;
}

fn check(rc: i32, operation: &str) -> Result<(), Error> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("{operation}: Native rc={rc}").into())
    }
}
fn unhex(text: &str) -> Result<Vec<u8>, Error> {
    if !text.len().is_multiple_of(2) || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("odd hexadecimal length".into());
    }
    (0..text.len())
        .step_by(2)
        .map(|i| Ok(u8::from_str_radix(&text[i..i + 2], 16)?))
        .collect()
}
fn id(text: &str) -> Result<Id, Error> {
    let bytes = unhex(text)?;
    if bytes.len() != 16 {
        return Err("expected a 128-bit Native ID".into());
    }
    Ok(Id([
        u64::from_ne_bytes(bytes[..8].try_into()?),
        u64::from_ne_bytes(bytes[8..].try_into()?),
    ]))
}
fn ids(values: &[Id]) -> String {
    format!(
        "{{{}}}",
        values
            .iter()
            .map(|&value| hex(value))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn message(text: *const c_char) -> String {
    unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned()
}

struct ResultSet(*mut c_void);
impl Drop for ResultSet {
    fn drop(&mut self) {
        unsafe { PQclear(self.0) };
    }
}
impl ResultSet {
    fn rows(&self) -> usize {
        unsafe { PQntuples(self.0) as usize }
    }
    fn value(&self, row: usize, column: usize) -> String {
        message(unsafe { PQgetvalue(self.0, row as i32, column as i32) })
    }
}
struct Database {
    connection: *mut c_void,
    commands: usize,
    committed: bool,
}
impl Drop for Database {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.query("ROLLBACK", &[]);
        }
        unsafe { PQfinish(self.connection) };
    }
}
impl Database {
    fn connect() -> Result<Self, Error> {
        let conninfo = CString::new(std::env::var("LAPLACE_CONNINFO")?)?;
        let connection = unsafe { PQconnectdb(conninfo.as_ptr()) };
        if connection.is_null() {
            return Err("libpq allocation failed".into());
        }
        let mut db = Self {
            connection,
            commands: 0,
            committed: false,
        };
        if unsafe { PQstatus(connection) } != 0 {
            return Err(message(unsafe { PQerrorMessage(connection) }).into());
        }
        db.query("BEGIN", &[])?;
        Ok(db)
    }
    fn query(&mut self, sql: &str, parameters: &[String]) -> Result<ResultSet, Error> {
        let sql = CString::new(sql)?;
        let values: Vec<_> = parameters
            .iter()
            .map(|value| CString::new(value.as_str()))
            .collect::<Result<_, _>>()?;
        let pointers: Vec<_> = values.iter().map(|value| value.as_ptr()).collect();
        let result = unsafe {
            PQexecParams(
                self.connection,
                sql.as_ptr(),
                pointers.len() as i32,
                ptr::null(),
                pointers.as_ptr(),
                ptr::null(),
                ptr::null(),
                0,
            )
        };
        self.commands += 1;
        if result.is_null() {
            return Err(message(unsafe { PQerrorMessage(self.connection) }).into());
        }
        let result = ResultSet(result);
        match unsafe { PQresultStatus(result.0) } {
            1 | 2 | 4 => Ok(result),
            _ => Err(message(unsafe { PQresultErrorMessage(result.0) }).into()),
        }
    }
    fn paths(&mut self, values: &[Id]) -> Result<HashMap<Id, Vec<Vertex>>, Error> {
        let result = self.query(
            "SELECT entity::text,encode(ST_AsEWKB(path),'hex') FROM laplace_paths($1::blake3[])",
            &[ids(values)],
        )?;
        let mut paths = HashMap::new();
        for row in 0..result.rows() {
            let bytes = unhex(&result.value(row, 1))?;
            let count =
                unsafe { lp_path_vertices(bytes.as_ptr(), bytes.len(), ptr::null_mut(), 0) };
            if count == 0 {
                return Err("stored path has no Native vertices".into());
            }
            let mut vertices = vec![Vertex::default(); count];
            if unsafe {
                lp_path_vertices(bytes.as_ptr(), bytes.len(), vertices.as_mut_ptr(), count)
            } != count
            {
                return Err("stored path cardinality changed during Native decode".into());
            }
            if paths.insert(id(&result.value(row, 0))?, vertices).is_some() {
                return Err("duplicate stored path".into());
            }
        }
        Ok(paths)
    }
    fn copy(&mut self, table: &str, columns: &str, bytes: &[u8]) -> Result<(), Error> {
        let result = self.query(&format!("COPY {table} ({columns}) FROM STDIN BINARY"), &[])?;
        if unsafe { PQresultStatus(result.0) } != 4 {
            return Err("COPY did not start".into());
        }
        drop(result);
        for chunk in bytes.chunks(1024 * 1024) {
            if unsafe { PQputCopyData(self.connection, chunk.as_ptr().cast(), chunk.len() as i32) }
                != 1
            {
                return Err(message(unsafe { PQerrorMessage(self.connection) }).into());
            }
        }
        if unsafe { PQputCopyEnd(self.connection, ptr::null()) } != 1 {
            return Err("COPY did not finish".into());
        }
        loop {
            let result = unsafe { PQgetResult(self.connection) };
            if result.is_null() {
                break;
            }
            let result = ResultSet(result);
            if unsafe { PQresultStatus(result.0) } != 1 {
                return Err(message(unsafe { PQresultErrorMessage(result.0) }).into());
            }
        }
        Ok(())
    }
}

struct Stage(*mut c_void);
impl Drop for Stage {
    fn drop(&mut self) {
        unsafe { canonical_stage_free(self.0) };
    }
}
impl Stage {
    fn new() -> Result<Self, Error> {
        let stage =
            unsafe { canonical_stage_new(2_000_000, 12_000_000, 512 * 1024 * 1024, ptr::null()) };
        if stage.is_null() {
            Err("Native canonical stage unavailable".into())
        } else {
            Ok(Self(stage))
        }
    }
    fn text(&self, value: &str) -> Result<Reference, Error> {
        let mut out = Reference::default();
        check(
            unsafe { canonical_stage_text(self.0, value.as_ptr(), value.len(), &mut out) },
            "compose text",
        )?;
        Ok(out)
    }
    fn compose(&self, children: &[Reference]) -> Result<Reference, Error> {
        let tier = children
            .iter()
            .map(|child| child.tier)
            .max()
            .ok_or("empty composition")?
            .checked_add(1)
            .ok_or("tier overflow")?;
        let mut out = Reference::default();
        check(
            unsafe {
                canonical_stage_compose(
                    self.0,
                    children.as_ptr(),
                    children.len(),
                    tier,
                    0,
                    0,
                    0,
                    &mut out,
                )
            },
            "compose ordered content",
        )?;
        Ok(out)
    }
}
fn expanded(vertices: &[Vertex]) -> Vec<Id> {
    vertices
        .iter()
        .flat_map(|vertex| std::iter::repeat_n(vertex.id(), vertex.run.max(1) as usize))
        .collect()
}

pub(super) fn run(source: &Views, file_id: &str, receipt_path: &str) -> Result<(), Error> {
    let start = Instant::now();
    if std::mem::size_of::<Reference>() != 64 || std::mem::size_of::<Vertex>() != 36 {
        return Err("Native staging/path ABI mismatch".into());
    }
    for (index, piece) in source.pieces.iter().enumerate() {
        if piece.index as usize != index {
            return Err("vocabulary ordinals require dense source IDs".into());
        }
    }
    let mut db = Database::connect()?;
    let file_id = id(file_id)?;
    let file = db.query("SELECT e.tier,(ST_X(e.coord)*9007199254740992)::bigint,(ST_Y(e.coord)*9007199254740992)::bigint,(ST_Z(e.coord)*9007199254740992)::bigint,(ST_M(e.coord)*9007199254740992)::bigint FROM entity e JOIN physicality p ON p.entity=e.id WHERE e.id=$1::blake3",&[hex(file_id)])?;
    if file.rows() != 1 {
        return Err("original tokenizer file is not durably placed".into());
    }
    let mut file_ref = Reference {
        id: file_id,
        tier: file.value(0, 0).parse()?,
        ..Reference::default()
    };
    for axis in 0..4 {
        file_ref.coordinate[axis] = file.value(0, axis + 1).parse()?;
    }
    let original = db.paths(&[file_id])?;
    let vertices = original.get(&file_id).ok_or("original file path missing")?;
    if vertices.len() != 2 || vertices[0].said != 4 {
        return Err("original file has no metadata/content trunk".into());
    }
    let content = db.query("SELECT laplace_text($1::blake3)", &[hex(vertices[1].id())])?;
    if content.rows() != 1 || content.value(0, 0).as_bytes() != source.source_bytes {
        return Err("persisted tokenizer content differs from decoded input".into());
    }

    let stage = Stage::new()?;
    check(
        unsafe { canonical_stage_reference(stage.0, &file_ref) },
        "reference admitted source file",
    )?;
    let byte_tag = stage.text("Byte")?;
    let nfc = stage.text("NFC")?;
    let mut numbers = HashMap::new();
    let mut raw = Vec::with_capacity(source.pieces.len());
    let mut decoded = Vec::with_capacity(source.pieces.len());
    let mut byte_fragments = 0;
    for piece in &source.pieces {
        raw.push(stage.text(&piece.raw)?);
        if let Some(text) = &piece.text {
            decoded.push(stage.text(text)?);
        } else {
            byte_fragments += 1;
            let mut bytes = Vec::with_capacity(piece.bytes.len());
            for &value in &piece.bytes {
                if !numbers.contains_key(&value) {
                    numbers.insert(value, stage.text(&value.to_string())?);
                }
                bytes.push(numbers[&value]);
            }
            let values = stage.compose(&bytes)?;
            decoded.push(stage.compose(&[byte_tag, values])?);
        }
    }
    let raw_root = stage.compose(&raw)?;
    let decoded_root = stage.compose(&decoded)?;
    let root = stage.compose(&[file_ref, nfc, raw_root, decoded_root])?;
    check(
        unsafe { canonical_stage_begin(stage.0, &root.id, 1) },
        "begin Native subtree descent",
    )?;
    let mut rounds = 0;
    let mut probed = 0;
    loop {
        let mut count = 0;
        let frontier = unsafe { canonical_stage_frontier(stage.0, &mut count) };
        if count == 0 {
            break;
        }
        if frontier.is_null() {
            return Err("Native frontier pointer missing".into());
        }
        let frontier = unsafe { std::slice::from_raw_parts(frontier, count) };
        let result = db.query(
            "SELECT ordinal FROM laplace_present_ids($1::blake3[])",
            &[ids(frontier)],
        )?;
        let mut present = vec![0u8; count];
        for row in 0..result.rows() {
            let ordinal: usize = result.value(row, 0).parse()?;
            if ordinal == 0 || ordinal > count {
                return Err("presence ordinal outside frontier".into());
            }
            present[ordinal - 1] = 1;
        }
        check(
            unsafe { canonical_stage_answer(stage.0, present.as_ptr(), count) },
            "answer Native subtree descent",
        )?;
        rounds += 1;
        probed += count;
    }
    check(
        unsafe { canonical_stage_finish(stage.0) },
        "finish Native COPY payloads",
    )?;
    let mut payloads = Vec::new();
    let mut row_count = 0;
    for table in 1..=2 {
        let (mut bytes, mut rows) = (0, 0);
        let data = unsafe { canonical_stage_copy(stage.0, table, &mut bytes, &mut rows) };
        if data.is_null() {
            return Err("Native COPY stream missing".into());
        }
        payloads.push((table, unsafe { std::slice::from_raw_parts(data, bytes) }));
        if table == 1 {
            row_count = rows;
        } else if rows != row_count {
            return Err("unpaired Native COPY rows".into());
        }
    }
    let mut published = [0u64; 3];
    if row_count != 0 {
        db.query("SELECT laplace_ensure_stage()", &[])?;
        let transaction = db
            .query("SELECT pg_current_xact_id()::text", &[])?
            .value(0, 0);
        if !transaction.bytes().all(|b| b.is_ascii_digit()) {
            return Err("invalid transaction identity".into());
        }
        let entity = format!("stage.lp_entity_{transaction}");
        let physicality = format!("stage.lp_path_{transaction}");
        db.query(&format!("CREATE UNLOGGED TABLE {entity} (LIKE stage.entity INCLUDING DEFAULTS,present boolean NOT NULL DEFAULT false)"),&[])?;
        db.query(&format!("CREATE UNLOGGED TABLE {physicality} (LIKE stage.physicality INCLUDING DEFAULTS,present boolean NOT NULL DEFAULT false)"),&[])?;
        for (table, bytes) in payloads {
            let columns = message(unsafe { canonical_stage_columns(table) });
            db.copy(
                if table == 1 { &entity } else { &physicality },
                &columns,
                bytes,
            )?;
        }
        let result = db.query(
            "SELECT * FROM laplace_merge_stage($1::regclass,$2::regclass)",
            &[entity.clone(), physicality.clone()],
        )?;
        for (column, value) in published.iter_mut().enumerate() {
            *value = result.value(0, column).parse()?;
        }
        db.query(&format!("DROP TABLE {entity},{physicality}"), &[])?;
    }
    // Verify every source index against the persisted ordered paths. Native
    // expands runs; the sorted containment index is never used as sequence.
    let paths = db.paths(&[root.id, raw_root.id, decoded_root.id])?;
    let checks = [
        (
            root.id,
            vec![file_ref.id, nfc.id, raw_root.id, decoded_root.id],
        ),
        (raw_root.id, raw.iter().map(|value| value.id).collect()),
        (
            decoded_root.id,
            decoded.iter().map(|value| value.id).collect(),
        ),
    ];
    for (id, expected) in checks {
        let actual = paths.get(&id).ok_or("admitted vocabulary path missing")?;
        if expanded(actual) != expected {
            return Err("persisted vocabulary index/content binding differs".into());
        }
    }
    let database_time = db.query("SELECT clock_timestamp()::text,current_database()", &[])?;
    let observed_at = database_time.value(0, 0);
    let database = database_time.value(0, 1);
    db.query("COMMIT", &[])?;
    db.committed = true;
    let receipt = json!({"observedAt":observed_at,"database":database,"sourceFile":hex(file_id),
        "tokenizerResultRoot":hex(root.id),"rawVocabularyRoot":hex(raw_root.id),"decodedVocabularyRoot":hex(decoded_root.id),
        "sourceTokenIds":source.pieces.len(),"byteFragmentDescriptors":byte_fragments,
        "normalization":"NFC; surrounding whitespace and case retained; sequence decoder conditions remain in original source tree",
        "rootConstituents":["original tokenizer file trunk","NFC","raw vocabulary in source-index order","decoded vocabulary in source-index order"],
        "byteFragmentForm":"[Byte, ordered canonical decimal byte values]; source codec remains in original tokenizer tree",
        "nativeDescentRounds":rounds,"nativePresenceIdsProbed":probed,"nativePairedRowsCopied":row_count,
        "newEntityRows":published[0],"newPhysicalityRows":published[1],"updatedRows":published[2],
        "allRawAndDecodedOrdinalsVerifiedFromPostgres":true,"sqlCommands":db.commands,"seconds":start.elapsed().as_secs_f64()});
    fs::write(receipt_path, serde_json::to_string_pretty(&receipt)? + "\n")?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
