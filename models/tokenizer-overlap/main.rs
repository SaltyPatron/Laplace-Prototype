use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ffi::{CString, c_void},
    fs,
    time::Instant,
};
use tokenizers::Tokenizer;
use unicode_normalization::UnicodeNormalization;

#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
struct Id([u64; 2]);

#[link(name = "laplace_core")]
unsafe extern "C" {
    fn codepoint_table_load_perfcache(path: *const i8) -> i32;
    fn laplace_content_root_id(bytes: *const u8, len: usize, id: *mut Id) -> i32;
    fn laplace_content_word_segment(
        bytes: *const u8,
        len: usize,
        emit: unsafe extern "C" fn(*mut c_void, u32, *const u8, u32, *const Id),
        context: *mut c_void,
    ) -> i32;
}

type Error = Box<dyn std::error::Error + Send + Sync>;

#[derive(Default)]
struct Segments {
    all: HashSet<Id>,
    lexical: HashSet<Id>,
    ordinal: u32,
    invalid: bool,
}

unsafe extern "C" fn collect(
    context: *mut c_void,
    ordinal: u32,
    bytes: *const u8,
    len: u32,
    id: *const Id,
) {
    let result = unsafe { &mut *(context as *mut Segments) };
    if ordinal != result.ordinal {
        result.invalid = true;
        return;
    }
    result.ordinal += 1;
    let text = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    if let Ok(text) = std::str::from_utf8(text) {
        let id = unsafe { *id };
        result.all.insert(id);
        if text.chars().any(char::is_alphanumeric) {
            result.lexical.insert(id);
        }
    } else {
        result.invalid = true;
    }
}

fn identity(text: &str) -> Result<Id, Error> {
    let mut id = Id([0; 2]);
    let rc = unsafe { laplace_content_root_id(text.as_ptr(), text.len(), &mut id) };
    if rc != 0 {
        return Err(format!("Native identity rc={rc} for {text:?}").into());
    }
    Ok(id)
}

fn hex(id: Id) -> String {
    id.0.into_iter()
        .flat_map(u64::to_ne_bytes)
        .map(|b| format!("{b:02x}"))
        .collect()
}

// Strict inverse of the published ByteLevel alphabet. This guard preserves byte
// fragments that the provider decoder intentionally realizes through UTF-8 loss.
// All valid pieces are checked against the actual Hugging Face decoder below.
// https://github.com/huggingface/tokenizers/blob/v0.23.2/tokenizers/src/pre_tokenizers/byte_level.rs
fn byte_alphabet() -> HashMap<char, u8> {
    let mut bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xa1..=0xac)
        .chain(0xae..=0xff)
        .collect();
    let mut chars: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
    let mut extra = 256;
    for b in 0..=255u8 {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(extra);
            extra += 1;
        }
    }
    chars
        .into_iter()
        .zip(bytes)
        .map(|(c, b)| (char::from_u32(c).unwrap(), b))
        .collect()
}

#[derive(Default)]
struct Views {
    raw: HashSet<String>,
    bytes: HashSet<Vec<u8>>,
    fragments: HashSet<Vec<u8>>,
    exact: HashSet<Id>,
    neutral: HashSet<Id>,
    uax: HashSet<Id>,
    lexical: HashSet<Id>,
    surfaces: BTreeMap<String, Vec<u32>>,
    summary: Value,
}

fn model(path: &str, alphabet: &HashMap<char, u8>) -> Result<Views, Error> {
    let input = fs::read(path)?;
    let source: Value = serde_json::from_slice(&input)?;
    let decoder = &source["decoder"];
    let byte_level = decoder["type"] == "ByteLevel";
    let sequence = decoder["decoders"].as_array();
    let tiny_sequence = decoder["type"] == "Sequence"
        && sequence.is_some_and(|s| {
            s.len() == 4
                && s[0]["type"] == "Replace"
                && s[0]["pattern"]["String"] == "▁"
                && s[0]["content"] == " "
                && s[1]["type"] == "ByteFallback"
                && s[2]["type"] == "Fuse"
                && s[3]["type"] == "Strip"
                && s[3]["content"] == " "
                && s[3]["start"] == 1
                && s[3]["stop"] == 0
        });
    if !byte_level && !tiny_sequence {
        return Err("decoder needs an explicit strict-byte recipe".into());
    }
    let tokenizer = Tokenizer::from_bytes(&input)?;
    let vocabulary = tokenizer.get_vocab(true);
    let mut by_id = BTreeMap::new();
    for (raw, id) in vocabulary {
        if by_id.insert(id, raw).is_some() {
            return Err("duplicate source index".into());
        }
    }
    let mut expected: HashSet<u32> = source["model"]["vocab"]
        .as_object()
        .ok_or("object vocabulary required")?
        .values()
        .map(|v| {
            v.as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .ok_or("invalid source index")
        })
        .collect::<Result<_, _>>()?;
    let mut special = HashSet::new();
    for added in source["added_tokens"]
        .as_array()
        .ok_or("added_tokens array required")?
    {
        let id = u32::try_from(added["id"].as_u64().ok_or("invalid added-token id")?)?;
        expected.insert(id);
        if added["special"] == true {
            special.insert(id);
        }
    }
    if by_id.keys().copied().collect::<HashSet<_>>() != expected {
        return Err("provider omitted source IDs".into());
    }
    let anchor = tokenizer
        .token_to_id("king")
        .ok_or("source lacks the declared valid codec prefix")?;
    let prefix = tokenizer.decode(&[anchor], false)?;
    let mut result = Views::default();
    let (mut invalid_tokens, mut special_tokens, mut whitespace_tokens, mut checked) = (0, 0, 0, 0);
    let mut normalization_changes = 0;
    let mut fragment_examples = Vec::new();
    for (id, raw) in &by_id {
        result.raw.insert(raw.clone());
        let bytes = if byte_level {
            raw.chars()
                .map(|c| alphabet.get(&c).copied())
                .collect::<Option<Vec<_>>>()
                .unwrap_or_else(|| raw.as_bytes().to_vec())
        } else if raw.len() == 6 && raw.starts_with("<0x") && raw.ends_with('>') {
            vec![u8::from_str_radix(&raw[3..5], 16)?]
        } else {
            raw.replace('▁', " ").into_bytes()
        };
        result.bytes.insert(bytes.clone());
        // A valid prefix makes TinyLlama's first-position Strip rule apply to
        // that prefix, exposing an interior piece without dropping its space.
        // This is codec execution, with no model weights or model prompting.
        let decoded = tokenizer.decode(&[anchor, *id], false)?;
        let decoded = decoded
            .strip_prefix(&prefix)
            .ok_or("decoder changed the codec prefix")?;
        let Ok(text) = std::str::from_utf8(&bytes) else {
            invalid_tokens += 1;
            result.fragments.insert(bytes.clone());
            if !decoded.contains('\u{fffd}') {
                return Err(
                    "invalid bytes did not produce the provider's replacement realization".into(),
                );
            }
            if fragment_examples.len() < 6 {
                fragment_examples.push(json!({"id":id,"raw":raw,"bytes":bytes.iter().map(|b|format!("{b:02x}")).collect::<String>()}));
            }
            continue;
        };
        if decoded != text {
            return Err(format!("strict/provider decoding differs at {id}: {raw:?}").into());
        }
        checked += 1;
        let nfc: String = text.nfc().collect();
        if nfc != text {
            normalization_changes += 1;
        }
        if !nfc.is_empty() {
            result.exact.insert(identity(&nfc)?);
        }
        if special.contains(id) {
            special_tokens += 1;
            continue;
        }
        let neutral = nfc.trim();
        if neutral.is_empty() {
            whitespace_tokens += 1;
            continue;
        }
        result.neutral.insert(identity(neutral)?);
        result
            .surfaces
            .entry(neutral.to_owned())
            .or_default()
            .push(*id);
        let mut segments = Segments::default();
        let rc = unsafe {
            laplace_content_word_segment(
                nfc.as_ptr(),
                nfc.len(),
                collect,
                &mut segments as *mut _ as *mut c_void,
            )
        };
        if rc != 0 || segments.invalid {
            return Err(format!("Native UAX segments rc={rc}").into());
        }
        result.uax.extend(segments.all);
        result.lexical.extend(segments.lexical);
    }
    result.summary = json!({"path":path,"sourceTokenIds":by_id.len(),"completeSourceIdCoverage":true,"specialTokenIds":special_tokens,
        "validUtf8TokensCheckedAgainstProvider":checked,"incompleteOrInvalidUtf8TokenIds":invalid_tokens,
        "distinctIncompleteOrInvalidByteSequences":result.fragments.len(),"whitespaceOnlyNonSpecialIds":whitespace_tokens,
        "nfcChangedTokenIds":normalization_changes,"fragmentExamples":fragment_examples,"decoder":decoder});
    Ok(result)
}

fn metric<T: Eq + std::hash::Hash>(a: &HashSet<T>, b: &HashSet<T>) -> Value {
    let shared = a.intersection(b).count();
    let union = a.len() + b.len() - shared;
    json!({"tinyDistinct":a.len(),"phiDistinct":b.len(),"shared":shared,"union":union,
        "tinyOnly":a.len()-shared,"phiOnly":b.len()-shared,
        "percentOfTiny":100.0*shared as f64/a.len().max(1) as f64,
        "percentOfPhi":100.0*shared as f64/b.len().max(1) as f64,
        "percentOfUnion":100.0*shared as f64/union.max(1) as f64})
}

fn main() -> Result<(), Error> {
    let start = Instant::now();
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err(
            "usage: laplace-tokenizer-overlap TINY_TOKENIZER PHI_TOKENIZER RECEIPT.json".into(),
        );
    }
    let floor = CString::new(std::env::var("LAPLACE_TIER0")?)?;
    let rc = unsafe { codepoint_table_load_perfcache(floor.as_ptr()) };
    if rc != 0 {
        return Err(format!("canonical Native perfcache load rc={rc}").into());
    }
    let alphabet = byte_alphabet();
    let tiny = model(&args[1], &alphabet)?;
    let phi = model(&args[2], &alphabet)?;
    let probes: Vec<_> = ["king","queen","some","thing","something","ing","bringing","singing","£","é","_","Paris"]
        .iter().map(|text|Ok(json!({"content":text,"nativeId":hex(identity(text)?),"tinyIds":tiny.surfaces.get(*text),"phiIds":phi.surfaces.get(*text)})))
        .collect::<Result<_, Error>>()?;
    let output = json!({"provider":"Hugging Face tokenizers 0.23.2; declared decoder; interior-piece codec comparison",
        "identity":"selected Laplace Native canonical content IDs; NFC; case preserved",
        "measurementSeconds":start.elapsed().as_secs_f64(),"models":[tiny.summary,phi.summary],
        "metrics":{
            "rawSourceSpellings":metric(&tiny.raw,&phi.raw),
            "exactDecodedByteSequences":metric(&tiny.bytes,&phi.bytes),
            "incompleteOrInvalidByteSequences":metric(&tiny.fragments,&phi.fragments),
            "exactNfcTextIncludingBoundaryWhitespaceAndSpecials":metric(&tiny.exact,&phi.exact),
            "boundaryNeutralNfcTextExcludingSpecialsAndEmpty":metric(&tiny.neutral,&phi.neutral),
            "nativeUaxSegmentsIncludingPunctuation":metric(&tiny.uax,&phi.uax),
            "nativeUaxSegmentsContainingLettersOrNumbers":metric(&tiny.lexical,&phi.lexical)},
        "examples":probes});
    fs::write(&args[3], serde_json::to_string_pretty(&output)? + "\n")?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
