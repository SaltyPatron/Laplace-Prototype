using System.Collections;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Laplace.Decomposers.Abstractions;
using Laplace.Decomposers.Model;
using Laplace.Engine.Core;
using Laplace.SubstrateCRUD;
using Npgsql;

// Executable calculation-to-record proof. Source components are evaluated, not
// prompted for answers. This deliberately does not claim complete model ETL.
static class ModelRecords
{
    const BindingFlags Methods=BindingFlags.Public|BindingFlags.NonPublic|BindingFlags.Instance|BindingFlags.Static;
    const string Contract="isolated FFN on source embedding; source readout; native circuit arena calibration fp1e9; complete declared candidate scope; v1";
    static readonly JsonSerializerOptions Json=new(){WriteIndented=true};
    [StructLayout(LayoutKind.Sequential)]
    struct Component {
        public Hash128 Id;public double X,Y,Z,M;public uint Atom;public byte Tier,HasAtom;public ushort Padding;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct Result {
        public Hash128 Id;public double X,Y,Z,M;public Hilbert128 Hilbert;public byte Tier;
        public byte P1,P2,P3,P4,P5,P6,P7;public nuint FirstPhysicality,EmittedPhysicalities;
    }
    [DllImport("laplace-model-components",EntryPoint="lp_model_conditional_record",CallingConvention=CallingConvention.Cdecl)]
    static extern int Record(IntentStage stage,[MarshalAs(UnmanagedType.LPUTF8Str)]string contract,
        [MarshalAs(UnmanagedType.LPUTF8Str)]string relation,[In]Component[] condition,nuint conditionCount,
        [In]Component[] objects,[In]long[] scores,nuint count,ref Hash128 source,long observed,
        out Result input,out Result response,out Result record,[Out]byte[] error,nuint errorBytes);
    [DllImport("laplace_core",EntryPoint="laplace_walk_edge_weight",CallingConvention=CallingConvention.Cdecl)]
    static extern double WalkWeight(long rating,long rd);
    static string Hex(Hash128 id)=>Convert.ToHexStringLower(id.ToBytes());
    static object Get(object value,string name)=>value.GetType().GetProperty(name,Methods)!.GetValue(value)!;
    static object Call(object value,string name,params object?[] args)=>value.GetType().GetMethod(name,Methods)!.Invoke(value,args)!;
    static void Require(bool condition,string message){if(!condition)throw new InvalidDataException(message);}
    static void Reject(Action action,string message) {
        bool refused=false;try{action();}catch(InvalidDataException){refused=true;}
        Require(refused,message);
    }
    static Hash128[] Vocabulary(NpgsqlConnection db,string root,string package)=>
        (Hash128[])typeof(Program).GetMethod("ReadVocabulary",Methods)!.Invoke(null,[db,root,File.ReadAllBytes(Path.Combine(package,"tokenizer.json"))])!;
    static void Save(string path,object value)=>File.WriteAllText(path,JsonSerializer.Serialize(value,Json)+"\n");

    sealed class Store : IDisposable
    {
        readonly NpgsqlConnection db;readonly NpgsqlTransaction tx;
        readonly Dictionary<(Hash128,short),double[]> paths=new();
        public Store(string connection) {
            db=new(connection);db.Open();tx=db.BeginTransaction();
            Exec("SET LOCAL statement_timeout='60s'; SET LOCAL lock_timeout='5s'");
            foreach(var table in new[]{"entities","physicalities","attestations"})
                Exec($"CREATE TEMP TABLE model_record_{table} (LIKE laplace.{table} INCLUDING DEFAULTS) ON COMMIT DROP");
            Exec("CREATE UNIQUE INDEX ON model_record_entities(id); CREATE UNIQUE INDEX ON model_record_physicalities(id); CREATE UNIQUE INDEX ON model_record_attestations(id)");
            Exec("CREATE INDEX model_record_condition_source ON model_record_attestations(object_id,source_id,context_id)");
        }
        public void Exec(string sql){using var q=new NpgsqlCommand(sql,db,tx);q.ExecuteNonQuery();}
        public long[] Counts() {
            using var q=new NpgsqlCommand("SELECT (SELECT count(*) FROM model_record_entities),(SELECT count(*) FROM model_record_physicalities),(SELECT count(*) FROM model_record_attestations)",db,tx);
            using var r=q.ExecuteReader();r.Read();return [r.GetInt64(0),r.GetInt64(1),r.GetInt64(2)];
        }
        public long[] Add(IntentStage stage) {
            long[] before=Counts();
            foreach(var item in new[]{(IntentStageTable.Entities,"entities"),(IntentStageTable.Physicalities,"physicalities"),(IntentStageTable.Attestations,"attestations")}) {
                string columns=IntentStage.CopyColumnList(item.Item1);byte[] sent=stage.EmitCopyBinary(item.Item1);
                Exec($"CREATE TEMP TABLE model_record_incoming (LIKE model_record_{item.Item2} INCLUDING DEFAULTS) ON COMMIT DROP");
                using(var copy=db.BeginRawBinaryCopy($"COPY model_record_incoming ({columns}) FROM STDIN (FORMAT BINARY)"))copy.Write(sent);
                using(var copy=db.BeginRawBinaryCopy($"COPY model_record_incoming ({columns}) TO STDOUT (FORMAT BINARY)")) {
                    using var bytes=new MemoryStream();copy.CopyTo(bytes);
                    Require(sent.AsSpan().SequenceEqual(bytes.ToArray()),"PostgreSQL changed native COPY bytes for "+item.Item2);
                }
                Exec($"INSERT INTO model_record_{item.Item2} SELECT DISTINCT ON(id) * FROM model_record_incoming ORDER BY id ON CONFLICT(id) DO NOTHING");
                Exec("DROP TABLE model_record_incoming");
            }
            long[] after=Counts();return after.Zip(before,(a,b)=>a-b).ToArray();
        }
        void ReadPaths(IEnumerable<Hash128> ids) {
            byte[][] missing=ids.Distinct().Where(id=>!paths.ContainsKey((id,1))).Select(id=>id.ToBytes()).ToArray();
            if(missing.Length==0)return;
            using var q=new NpgsqlCommand("SELECT entity_id,type,generation.trajectory_bits(ST_AsBinary(trajectory,'NDR')) FROM model_record_physicalities WHERE entity_id=ANY($1) AND trajectory IS NOT NULL",db,tx);
            q.Parameters.AddWithValue(missing);using var reader=q.ExecuteReader();
            while(reader.Read()) {
                var bytes=reader.GetFieldValue<byte[]>(2);var result=new double[bytes.Length/8];Buffer.BlockCopy(bytes,0,result,0,bytes.Length);
                paths[(Hash128.FromBytes(reader.GetFieldValue<byte[]>(0)),reader.GetInt16(1))]=result;
            }
        }
        double[] Path(Hash128 id,short type)=>paths.TryGetValue((id,type),out var result)?result:throw new InvalidDataException("stored result body missing: "+Hex(id));
        public void Verify(Result record,Result input,Result response,Component[] candidates,long[] scores) {
            // Every verification reads current stored bodies, including after a
            // synthetic-control savepoint has been rolled back.
            paths.Clear();
            ReadPaths([record.Id,response.Id]);
            Hash128[] fields=Trajectory.Constituents(Path(record.Id,1));
            Require(fields.Length==4&&fields[1]==input.Id&&fields[3]==response.Id,"conditional record fields changed");
            Hash128[] pairs=Trajectory.Constituents(Path(response.Id,1));
            Require(pairs.Length==candidates.Length,"conditional candidate scope changed");
            var walk=TestimonyWalk.Unpack(Path(record.Id,3));
            Require(walk.Length==pairs.Length,"query carrier disagrees with content scope");
            var expected=candidates.Select((c,i)=>(c.Id,score:scores[i])).ToDictionary(x=>x.Id,x=>x.score);
            var seen=new HashSet<Hash128>();
            ReadPaths(pairs);
            for(int i=0;i<pairs.Length;i++) {
                Hash128[] pair=Trajectory.Constituents(Path(pairs[i],1));
                Require(pair.Length==2&&seen.Add(pair[0])&&expected.ContainsKey(pair[0]),"response object missing or repeated");
                long score=expected[pair[0]];
                Require(walk[i].ObjectId==pair[0]&&walk[i].ScoreFp1e9==score,"projection lost a conditional grade");
                Require(pair[1]==ModelCoordinates.ScalarId(score.ToString(CultureInfo.InvariantCulture)),"grade text identity disagrees with packed grade");
            }
        }
        public object Attribution(Hash128 result) {
            using var q=new NpgsqlCommand("SELECT count(*),count(DISTINCT source_id),count(DISTINCT context_id) FROM model_record_attestations WHERE subject_id=$1",db,tx);
            q.Parameters.AddWithValue(result.ToBytes());using var r=q.ExecuteReader();r.Read();
            return new{receipts=r.GetInt64(0),sources=r.GetInt64(1),contexts=r.GetInt64(2)};
        }
        public object SourceReadback(Hash128 input,Hash128 source,int expected) {
            using var q=new NpgsqlCommand("SELECT count(*) FROM model_record_attestations WHERE object_id=$1 AND source_id=$2",db,tx);
            q.Parameters.AddWithValue(input.ToBytes());q.Parameters.AddWithValue(source.ToBytes());
            long count=(long)q.ExecuteScalar()!;Require(count==expected,"source-scoped condition readback lost or mixed records");
            using var explain=new NpgsqlCommand("EXPLAIN (FORMAT JSON) SELECT subject_id,context_id FROM model_record_attestations WHERE object_id=$1 AND source_id=$2",db,tx);
            explain.Parameters.AddWithValue(input.ToBytes());explain.Parameters.AddWithValue(source.ToBytes());
            return new{records=count,condition=Hex(input),source=Hex(source),plan=JsonSerializer.Deserialize<JsonElement>((string)explain.ExecuteScalar()!)};
        }
        public object ExistingCanonicalCoverage() {
            using var q=new NpgsqlCommand("SELECT count(*),count(e.id) FROM model_record_entities r LEFT JOIN laplace.entities e ON e.id=r.id",db,tx);
            using var r=q.ExecuteReader();r.Read();long total=r.GetInt64(0),known=r.GetInt64(1);
            return new{database=db.Database,recordedEntityIds=total,alreadyPresentEntityIds=known,absentEntityIds=total-known,scope="Exact canonical entity ID presence in this database at proof time; this does not compare physicality bodies or other databases."};
        }
        public object FoldStoredResponse(Result result) {
            var response=TestimonyWalk.Unpack(Path(result.Id,3))[0];
            var opponent=NativeAttestation.WitnessParameters(ModelDecomposer.CompletesToTypeId,SourceTrust.AiModelProbe);
            var state=Glicko2.Init(Glicko2.DefaultRatingFp1e9,Glicko2.DefaultRdFp1e9,Glicko2.DefaultVolatilityFp1e9);
            double before=WalkWeight(state.RatingFp1e9,state.RdFp1e9);
            Glicko2.FoldUniformPeriod(ref state,opponent.Rating,opponent.Rd,1,response.ScoreFp1e9,Glicko2.DefaultTauFp1e9,0);
            return new{objectId=Hex(response.ObjectId),gradeReadFromPostgres=response.ScoreFp1e9,ratingFp1e9=state.RatingFp1e9,rdFp1e9=state.RdFp1e9,walkTensionBefore=before,walkTensionAfter=WalkWeight(state.RatingFp1e9,state.RdFp1e9),scope="Native fold demonstration of this calculated coupling grade; no semantic truth claim or consensus write."};
        }
        public void Dispose(){tx.Rollback();tx.Dispose();db.Dispose();}
    }

    static (IntentStage Stage,Result Input,Result Response,Result Record) Lower(
        Component[] condition,Component[] candidates,long[] scores,Hash128 source,Hash128 context) {
        Require(candidates.Length==scores.Length,"response objects and grades must align");
        var stage=IntentStage.New(Math.Max(64,candidates.Length*8));
        try {
            byte[] error=new byte[1024];long observed=DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()*1000;
            int rc=Record(stage,Contract,"COMPLETES_TO",condition,(nuint)condition.Length,candidates,scores,(nuint)candidates.Length,
                ref source,observed,out var input,out var response,out var result,error,(nuint)error.Length);
            Require(rc==0,Encoding.UTF8.GetString(error).TrimEnd('\0'));
            stage.RecordPhysicalitySourceSince(0,source);
            var receipt=NativeAttestation.CategoricalResolved(result.Id,RelationTypeRegistry.RelationTypeId("DERIVED_FROM"),input.Id,source,context,SourceTrust.AiModelProbe);
            stage.AddAttestation(receipt.Id,receipt.SubjectId,receipt.TypeId,receipt.ObjectId,receipt.SourceId,receipt.ContextId,
                (short)receipt.Outcome,observed,receipt.ObservationCount,receipt.SumScoreFp1e9??receipt.ScoreFp1e9,
                receipt.OpponentRdFp1e9,receipt.QualifierMask|CalculationSources.Qualifier,receipt.OpponentRatingFp1e9);
            return(stage,input,response,result);
        }catch{stage.Dispose();throw;}
    }

    static int Main(string[] args) {
        var watch=Stopwatch.StartNew();string output=args.Length>0?args[0]:"";
        try {
            Require(args.Length==5,"usage: model-records OUTPUT_DIR PACKAGE_A BINDING_A PACKAGE_B BINDING_B; connections from LAPLACE_CONNINFO and LAPLACE_MONO_CONNINFO");
            Require(Marshal.SizeOf<Component>()==56&&Marshal.SizeOf<Result>()==88,"native record ABI mismatch");
            Directory.CreateDirectory(output);
            var runtime=Directory.EnumerateFiles(AppContext.BaseDirectory).Where(p=>Path.GetFileName(p) is "model-records.dll" or "Laplace.Decomposers.dll" or "Laplace.Core.dll" or "Laplace.Substrate.dll" or "liblaplace_core.so" or "liblaplace_dynamics.so" or "liblaplace-model-components.so")
                .ToDictionary(p=>p,p=>Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(p))));
            CodepointPerfcache.Load(Environment.GetEnvironmentVariable("LAPLACE_TIER0")??throw new InvalidOperationException("LAPLACE_TIER0 required"));
            using var bindings=new NpgsqlConnection(Environment.GetEnvironmentVariable("LAPLACE_CONNINFO")??throw new InvalidOperationException("LAPLACE_CONNINFO required"));bindings.Open();
            string[] packages=[args[1],args[3]],roots=[args[2],args[4]];
            Hash128[][] vocabulary=[Vocabulary(bindings,roots[0],packages[0]),Vocabulary(bindings,roots[1],packages[1])];
            var inSecond=vocabulary[1].ToHashSet();
            Hash128[] shared=vocabulary[0].Skip(512).Where(inSecond.Contains).Distinct().Take(128).ToArray();
            Require(shared.Length==128,"128 shared source-qualified candidates required");
            var components=new Dictionary<Hash128,Component>();var spellings=new Dictionary<Hash128,string>();
            using(var q=new NpgsqlCommand("SELECT id::text,tier,ST_X(coord),ST_Y(coord),ST_Z(coord),ST_M(coord),laplace_text(id) FROM entity WHERE id=ANY($1::text[]::blake3[])",bindings)) {
                q.Parameters.AddWithValue(shared.Select(Hex).ToArray());using var r=q.ExecuteReader();
                while(r.Read()) {
                    Hash128 id=Hash128.FromBytes(Convert.FromHexString(r.GetString(0)));byte tier=checked((byte)r.GetInt16(1));
                    string text=r.GetString(6);uint atom=0;
                    if(tier==0) {
                        Require(Rune.TryGetRuneAt(text,0,out var rune)&&text.Length==rune.Utf16SequenceLength,"tier-0 binding is not a Unicode scalar");
                        atom=checked((uint)rune.Value);
                    }
                    components[id]=new(){Id=id,Tier=tier,X=r.GetDouble(2),Y=r.GetDouble(3),Z=r.GetDouble(4),M=r.GetDouble(5),Atom=atom,HasAtom=(byte)(tier==0?1:0)};
                    spellings[id]=text;
                }
            }
            Require(components.Count==128,"shared canonical placements missing");
            Component[] candidates=shared.Select(id=>components[id]).ToArray();
            Save(Path.Combine(output,"bindings.json"),shared.Select(id=>new{entity=Hex(id),text=spellings[id],sourceIndexes=vocabulary.Select(v=>Array.IndexOf(v,id))}));
            using var store=new Store(Environment.GetEnvironmentVariable("LAPLACE_MONO_CONNINFO")??throw new InvalidOperationException("LAPLACE_MONO_CONNINFO required"));
            var rows=new List<object>();var modelCounts=new List<object>();var recordIds=new HashSet<Hash128>();object? controls=null;
            var estateType=typeof(ModelTokenEdgeETL).Assembly.GetType("Laplace.Decomposers.Model.ModelCircuitEstate",true)!;
            for(int model=0;model<2;model++) {
                using var snapshot=SourceEntityIdConventions.OpenModelContentSnapshot(packages[model])??throw new InvalidDataException("held source missing");
                var headers=SafetensorsContainerParser.ParseModel(snapshot);
                var config=ModelConfigReader.Read(Path.Combine(packages[model],"config.json"));
                var manifest=ModelManifest.Recognize(headers,config,vocabulary[model].Length,packages[model]);
                var tokens=shared.Select(id=>{var c=components[id];return new LlamaTokenizerParser.TokenRecord {
                    TokenId=Array.IndexOf(vocabulary[model],id),RawToken=spellings[id],CanonicalBytes=Encoding.UTF8.GetBytes(spellings[id]),
                    EntityId=id,Tier=c.Tier,IsByteLevel=false,Role=TokenRole.None,ContentX=c.X,ContentY=c.Y,ContentZ=c.Z,ContentM=c.M,HasContentCoord=true};}).ToArray();
                var map=shared.Select((id,i)=>(id,i)).ToDictionary(x=>x.id,x=>x.i);
                var selected=new SelectedModelAnalysisInput(packages[model],manifest,tokens,snapshot.SourceId,snapshot);
                var estate=Activator.CreateInstance(estateType,Methods,null,[selected,map],null)!;
                var iterator=((IEnumerable)Call(estate,"Enumerate",ModelDecomposer.CompletesToTypeId)).GetEnumerator();
                long[] before=store.Counts();int n=0;long units=0;double lowerSeconds=0,databaseSeconds=0;
                try {
                    while(iterator.MoveNext()) {
                        object descriptor=iterator.Current;
                        if((string)Get(descriptor,"Plane")!="ffn")continue;
                        int layer=(int)Get(descriptor,"Layer");object circuit=Get(descriptor,"Contraction");
                        var grades=((long[],short[]))Call(circuit,"Score",new int[shared.Length],Enumerable.Range(0,shared.Length).ToArray());
                        Hash128 context=(Hash128)Get(descriptor,"ContextId");
                        long lowerStart=Stopwatch.GetTimestamp();
                        var lowered=Lower([candidates[0]],candidates,grades.Item1,snapshot.SourceId,context);
                        double lowerTime=Stopwatch.GetElapsedTime(lowerStart).TotalSeconds;lowerSeconds+=lowerTime;
                        using(lowered.Stage) {
                            long dbStart=Stopwatch.GetTimestamp();
                            long[] delta=store.Add(lowered.Stage);store.Verify(lowered.Record,lowered.Input,lowered.Response,candidates,grades.Item1);
                            Require(store.Add(lowered.Stage).All(x=>x==0),"identical record replay added rows");
                            double dbTime=Stopwatch.GetElapsedTime(dbStart).TotalSeconds;databaseSeconds+=dbTime;
                            recordIds.Add(lowered.Record.Id);n++;
                            units+=manifest.Single(layer,TensorRoleKind.MlpUp)!.Shape[0];
                            rows.Add(new{model,layer,source=Hex(snapshot.SourceId),sourceContext=Hex(context),input=Hex(lowered.Input.Id),response=Hex(lowered.Response.Id),record=Hex(lowered.Record.Id),newEntities=delta[0],newPhysicalities=delta[1],newSourceReceipts=delta[2],recordLoweringSeconds=lowerTime,databaseRoundtripReadbackReplaySeconds=dbTime,nativeFold=store.FoldStoredResponse(lowered.Record)});
                            if(model==0&&layer==0) {
                                store.Exec("SAVEPOINT synthetic_controls");
                                Hash128 controlSource=Hash128.Text("explicit duplicate-source control; not a model observation");
                                var duplicate=Lower([candidates[0]],candidates,grades.Item1,controlSource,context);
                                using(duplicate.Stage) {
                                    Require(duplicate.Record.Id==lowered.Record.Id,"source leaked into canonical result identity");
                                    long[] distinct=store.Add(duplicate.Stage);
                                    Require(distinct[0]==0&&distinct[1]==0&&distinct[2]==1,"second source did not share content and retain its receipt");
                                    long[] changed=(long[])grades.Item1.Clone();changed[0]=changed[0]==1_000_000_000?changed[0]-1:changed[0]+1;
                                    var mutation=Lower([candidates[0]],candidates,changed,controlSource,context);
                                    using(mutation.Stage)Require(mutation.Record.Id!=lowered.Record.Id,"changed derived grade collapsed into the same content");
                                    var reordered=Lower([candidates[1],candidates[0]],candidates,grades.Item1,controlSource,context);
                                    using(reordered.Stage)Require(reordered.Record.Id!=lowered.Record.Id,"changed input condition collapsed into the same record");
                                    var permuted=Lower([candidates[0]],candidates.Reverse().ToArray(),grades.Item1.Reverse().ToArray(),controlSource,context);
                                    using(permuted.Stage)Require(permuted.Record.Id==lowered.Record.Id,"source candidate enumeration leaked into canonical result identity");
                                    var reversed=Lower([candidates[0],candidates[1]],candidates,grades.Item1,controlSource,context);
                                    using(reversed.Stage)Require(reversed.Record.Id!=reordered.Record.Id,"input condition order was lost");
                                    Reject(()=>{var invalid=Lower([candidates[0]],[candidates[0],candidates[0]],[0,1],controlSource,context);invalid.Stage.Dispose();},"duplicate canonical candidates accepted");
                                    Reject(()=>{var invalid=Lower([candidates[0]],[candidates[0]],[-1],controlSource,context);invalid.Stage.Dispose();},"out-of-domain grade accepted");
                                    var boundary=Lower([candidates[0]],[candidates[0],candidates[1]],[0,1_000_000_000],controlSource,context);
                                    using(boundary.Stage){store.Add(boundary.Stage);store.Verify(boundary.Record,boundary.Input,boundary.Response,[candidates[0],candidates[1]],[0,1_000_000_000]);}
                                    controls=new{identicalReplayNewRows=0,secondSourceNewEntities=distinct[0],secondSourceNewPhysicalities=distinct[1],secondSourceNewReceipt=distinct[2],attribution=store.Attribution(lowered.Record.Id),changedGradeDifferentRecord=true,changedConditionDifferentRecord=true,conditionOrderPreserved=true,candidatePermutationSameRecord=true,duplicateCandidatesRefused=true,outOfDomainGradeRefused=true,zeroAndOneGradesRoundTrip=true,controlSourceIsSynthetic=true,controlRowsExcludedFromModelCounts=true};
                                }
                                store.Exec("ROLLBACK TO SAVEPOINT synthetic_controls; RELEASE SAVEPOINT synthetic_controls");
                            }
                        }
                        Console.WriteLine($"model={model} layer={layer} distinct_results={recordIds.Count} elapsed_seconds={watch.Elapsed.TotalSeconds:F3}");
                    }
                }finally{(iterator as IDisposable)?.Dispose();}
                snapshot.VerifySourceId();long[] after=store.Counts();
                modelCounts.Add(new{model,package=packages[model],bindingRoot=roots[model],source=Hex(snapshot.SourceId),recognizedFfnUnits=units,calculatedFfnLayers=n,distinctConditionalRecordsCumulative=recordIds.Count,newRows=after.Zip(before,(a,b)=>a-b).ToArray(),cumulativeRows=after,recordLoweringSeconds=lowerSeconds,databaseRoundtripReadbackReplaySeconds=databaseSeconds,sourceReadback=store.SourceReadback(candidates[0].Id,snapshot.SourceId,n)});
            }
            foreach(var item in runtime)Require(Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(item.Key)))==item.Value,"runtime changed during proof");
            Save(Path.Combine(output,"result.json"),new{observedAt=DateTimeOffset.UtcNow,scope="All recognized dense FFN layers from two held real checkpoints, each calculated independently on one shared canonical input and the same complete declared 128-object response scope. No generated text or model API. This proves calculation lowering, not full learned-operator ingestion or contextual model execution.",contract=Contract,models=modelCounts,controls,records=rows,postgresRows=store.Counts(),existingCanonicalCoverage=store.ExistingCanonicalCoverage(),productionRuntimeSha256=runtime,elapsedSeconds=watch.Elapsed.TotalSeconds,postgresTransaction="rolled back on exit",canonicalDatabaseWrites=0,completeModelIngestionProved=false});
            Console.WriteLine($"verified=true models=2 distinct_conditional_results={recordIds.Count} elapsed_seconds={watch.Elapsed.TotalSeconds:F3}");return 0;
        }catch(Exception e){Console.Error.WriteLine(e);if(Directory.Exists(output))Save(Path.Combine(output,"failure.json"),new{at=DateTimeOffset.UtcNow,error=e.ToString(),elapsedSeconds=watch.Elapsed.TotalSeconds});return 1;}
    }
}
