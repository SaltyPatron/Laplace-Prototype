using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Laplace.Decomposers.Model;
using Laplace.Decomposers.Abstractions;
using Laplace.Engine.Core;
using Npgsql;

[StructLayout(LayoutKind.Sequential)]
struct TensorView {
    public int Fd, Dtype;
    public ulong Offset, Rows, Cols;
    public int Transpose, Reserved;
    public static TensorView Absent => new() { Fd=-1 };
}
[StructLayout(LayoutKind.Sequential)]
struct FfnBinding {
    public TensorView Embedding, Up, Gate, Down, UpBias, GateBias, DownBias;
    public int Activation, Layer;
}
[StructLayout(LayoutKind.Sequential)]
struct ComponentReceipt {
    public ulong Units, DecodedValues, ScratchBytes;
    public double MaximumError, RelativeError;
}
[StructLayout(LayoutKind.Sequential)]
struct AttentionBinding {
    public TensorView Embedding,Q,K,V,O,QBias,KBias,VBias,OBias,Norm,NormBias;
    public int Heads,KvHeads,HeadDim,RotaryDim,NormKind,Layer;
    public double Epsilon,RopeBase;
}
[StructLayout(LayoutKind.Sequential)]
struct AttentionReceipt {
    public ulong Heads,KvHeads,DecodedValues;
    public double ProjectionError,OutputError,PrefixError,PositionDelta,ContextDelta;
}
[StructLayout(LayoutKind.Sequential)]
struct NativePath { public IntPtr Vertices; public nuint Count; }
static class Program {
    [DllImport("laplace_core",EntryPoint="lp_path_of")]
    static extern NativePath PathOf(IntPtr bytes,nuint length);
    [DllImport("laplace_core",EntryPoint="lp_path_len")]
    static extern nuint PathLength(NativePath path);
    [DllImport("laplace_core",EntryPoint="lp_path_expand")]
    static extern nuint ExpandPath(NativePath path,[Out] Hash128[] ids,nuint capacity);
    [DllImport("laplace-model-components",EntryPoint="lp_model_ffn_components")]
    static extern int Components([In] FfnBinding[] layers,nuint count,ulong row,
        [Out] double[] results,nuint resultCount,[Out] ComponentReceipt[] receipts,
        [Out] byte[] error,nuint errorBytes);
    [DllImport("laplace-model-components",EntryPoint="lp_model_attention_components")]
    static extern int Attention([In] AttentionBinding[] layers,nuint count,[In] ulong[] rows,nuint sequence,
        [Out] double[] results,nuint resultCount,[Out] AttentionReceipt[] receipts,
        [Out] byte[] error,nuint errorBytes);

    static int Main(string[] args) {
        try {
            if(args.Length<4) throw new ArgumentException("usage: model-components TIER0 CONNINFO WORD[,WORD...] --binding-root ROOT PACKAGE; or WORD[,WORD...] --bindings ROOT");
            if(!OperatingSystem.IsLinux()) throw new PlatformNotSupportedException("the held-file view adapter currently uses Linux file descriptors");
            if(Marshal.SizeOf<TensorView>()!=40 || Marshal.SizeOf<FfnBinding>()!=288 || Marshal.SizeOf<ComponentReceipt>()!=40 ||
               Marshal.SizeOf<AttentionBinding>()!=480 || Marshal.SizeOf<AttentionReceipt>()!=64)
                throw new InvalidOperationException("native component ABI layout mismatch");
            CodepointPerfcache.Load(args[0]);
            using var db=new NpgsqlConnection(args[1]); db.Open();
            var words=args[2].Split(',',StringSplitOptions.TrimEntries);
            if(words.Any(string.IsNullOrEmpty))throw new ArgumentException("each comma-separated argument names one canonical content binding");
            using var query=new NpgsqlCommand("SELECT laplace_text_id(w)::text FROM unnest($1::text[]) WITH ORDINALITY AS x(w,i) ORDER BY i",db);
            query.Parameters.AddWithValue(words);
            var expected=new List<string>();
            using(var result=query.ExecuteReader())while(result.Read())expected.Add(result.GetString(0));
            var packages=args.Skip(3).ToList();
            if(packages.Count==2&&packages[0]=="--bindings") {
                var vocabulary=ReadVocabulary(db,packages[1],null);
                var rows=SelectRows(db,vocabulary,words,expected.ToArray());
                Console.WriteLine($"binding_only=true canonical_words={string.Join(',',words)} source_indexes={string.Join(',',rows)} model_source_files_opened=0");
                return 0;
            }
            if(packages.Count!=3||packages[0]!="--binding-root")
                throw new ArgumentException("component calculations require --binding-root ROOT PACKAGE, using an admitted tokenizer result");
            Run(packages[2],words,expected.ToArray(),db,packages[1]);
            return 0;
        } catch(Exception e) { Console.Error.WriteLine(e); return 1; }
    }

    static void Run(string directory,string[] words,string[] expected,NpgsqlConnection db,string bindingRoot) {
        var watch=Stopwatch.StartNew();
        string path=SafetensorSnapshotWitness.ResolveCompleteDir(directory)
            ?? throw new InvalidDataException("complete model package required");
        using var snapshot=SourceEntityIdConventions.OpenModelContentSnapshot(path)
            ?? throw new InvalidDataException("model source snapshot unavailable");
        var headers=SafetensorsContainerParser.ParseModel(snapshot);
        byte[] tokenizer=snapshot.Read(Path.Combine(path,"tokenizer.json"),Read);
        byte[] configBytes=snapshot.Read(Path.Combine(path,"config.json"),Read);
        var persisted=ReadVocabulary(db,bindingRoot,tokenizer);
        var config=ModelConfigReader.Read(configBytes);
        // Bind source-declared positional options not yet exposed by ModelConfig.
        using var sourceOptions=JsonDocument.Parse(configBytes);
        var manifest=ModelManifest.Recognize(headers,config,persisted.Length,path);
        var bound=manifest.Roles.SelectMany(r=>r.Bias is null?new[]{r.Name}:new[]{r.Name,r.Bias})
            .ToHashSet(StringComparer.Ordinal);
        var unbound=headers.Where(t=>!bound.Contains(t.Name)).Select(t=>t.Name).ToArray();
        if(unbound.Length!=0 || manifest.Anatomy.Conflicts.Count!=0 ||
           manifest.Anatomy.Operators.Any(o=>o.Slots.Any(s=>s.IsAmbiguous)))
            throw new InvalidDataException("source anatomy incomplete: "+string.Join(",",unbound.Concat(manifest.Anatomy.Conflicts)));
        if(manifest.Roles.Any(r=>r.Orientation=="square"))
            throw new InvalidDataException("a square projection requires an explicit source orientation");
        Console.WriteLine($"source_structure_bound_tensors={bound.Count} source_total_tensors={headers.Count}");
        foreach(var family in manifest.Roles.GroupBy(r=>r.Kind switch {
            TensorRoleKind.AttnQ or TensorRoleKind.AttnK or TensorRoleKind.AttnV or TensorRoleKind.AttnO or TensorRoleKind.AttnQkv=>"attention",
            TensorRoleKind.MlpUp or TensorRoleKind.MlpDown or TensorRoleKind.MlpGate or TensorRoleKind.MlpGateUp=>"ffn",
            TensorRoleKind.Norm=>"normalization",
            TensorRoleKind.Embedding or TensorRoleKind.LmHead=>"vocabulary",
            _=>r.Kind.ToString()
        })) {
            int count=family.SelectMany(r=>r.Bias is null?new[]{r.Name}:new[]{r.Name,r.Bias})
                .Distinct(StringComparer.Ordinal).Count();
            string calculation=family.Key=="ffn"?"connected-units":words.Length>1&&family.Key=="attention"?"connected-heads":
                words.Length>1&&family.Key=="normalization"?"attention-inputs":"source-binding";
            Console.WriteLine($"source_tensor_family={family.Key} tensor_count_including_bias={count} component_calculation={calculation}");
        }
        int[] sourceRows=SelectRows(db,persisted,words,expected);
        int tokenIndex=sourceRows[0];string word=words[0];
        string actual=expected[0];
        var byName=headers.ToDictionary(t=>t.Name,StringComparer.Ordinal);
        TensorView View(string? name,bool transpose=false) {
            if(name is null) return TensorView.Absent;
            var t=byName[name];
            int fd=snapshot.Read(t.FilePath,s=>((FileStream)s).SafeFileHandle.DangerousGetHandle().ToInt32());
            return new() { Fd=fd, Dtype=t.Dtype switch { "F32"=>0,"F16"=>1,"BF16"=>2,_=>throw new InvalidDataException(t.Dtype) },
                Offset=checked((ulong)t.AbsoluteDataStart),Rows=checked((ulong)t.Shape[0]),
                Cols=t.Shape.Length==1?1:checked((ulong)t.Shape[1]),Transpose=transpose?1:0 };
        }
        var embedding=manifest.Embedding ?? throw new InvalidDataException("recognized embedding required");
        var layers=new FfnBinding[manifest.LayerCount];
        for(int l=0;l<layers.Length;l++) {
            var up=manifest.Single(l,TensorRoleKind.MlpUp) ?? throw new InvalidDataException($"missing up binding {l}");
            var down=manifest.Single(l,TensorRoleKind.MlpDown) ?? throw new InvalidDataException($"missing down binding {l}");
            var gate=manifest.Single(l,TensorRoleKind.MlpGate);
            layers[l]=new() { Embedding=View(embedding.Name),Up=View(up.Name,!up.IsRowMajorProjection),
                Down=View(down.Name,!down.IsRowMajorProjection),Gate=gate is null?TensorView.Absent:View(gate.Name,!gate.IsRowMajorProjection),
                UpBias=View(up.Bias),GateBias=View(gate?.Bias),DownBias=View(down.Bias),
                Activation=manifest.FfnActivation(gate is not null) ?? throw new InvalidDataException("unsupported declared FFN activation"),Layer=l };
        }
        double bindingSeconds=watch.Elapsed.TotalSeconds;
        var receipts=new ComponentReceipt[layers.Length]; var error=new byte[1024];
        var results=new double[checked(layers.Length*manifest.Config.HiddenSize)];
        int rc=Components(layers,(nuint)layers.Length,checked((ulong)tokenIndex),results,(nuint)results.Length,receipts,error,(nuint)error.Length);
        if(rc!=0) throw new InvalidDataException(Encoding.UTF8.GetString(error).TrimEnd('\0'));
        if(words.Length>1) {
            if(!config.Config.Architecture.EndsWith("ForCausalLM",StringComparison.Ordinal) || config.Config.QkNorm ||
               manifest.Roles.Any(r=>r.Kind==TensorRoleKind.PositionEmbedding))
                throw new InvalidDataException("this attention calculation requires the declared causal RoPE layout without separate Q/K normalization");
            var options=sourceOptions.RootElement;
            if(options.TryGetProperty("rope_scaling",out var scaling)&&scaling.ValueKind!=JsonValueKind.Null)
                throw new InvalidDataException("scaled RoPE requires its declared position recipe");
            int normKind=options.TryGetProperty("rms_norm_eps",out var rms)&&rms.ValueKind==JsonValueKind.Number?0:
                options.TryGetProperty("layer_norm_eps",out var ln)&&ln.ValueKind==JsonValueKind.Number?1:
                throw new InvalidDataException("source must declare its input-normalization contract");
            double fraction=options.TryGetProperty("partial_rotary_factor",out var factor)?factor.GetDouble():1;
            var cfg=manifest.Config;double rotary=cfg.HeadDim*fraction;
            if(rotary!=Math.Truncate(rotary))throw new InvalidDataException("fractional rotary axis extent");
            var attention=new AttentionBinding[layers.Length];
            for(int l=0;l<attention.Length;l++) {
                TensorRole Role(TensorRoleKind kind)=>manifest.Single(l,kind)??throw new InvalidDataException($"missing attention role {kind} at {l}");
                var q=Role(TensorRoleKind.AttnQ);var k=Role(TensorRoleKind.AttnK);var v=Role(TensorRoleKind.AttnV);var o=Role(TensorRoleKind.AttnO);
                var norm=manifest.Norm(l,"attention")??throw new InvalidDataException("missing recognized attention input normalization");
                attention[l]=new() { Embedding=View(embedding.Name),Q=View(q.Name,!q.IsRowMajorProjection),K=View(k.Name,!k.IsRowMajorProjection),
                    V=View(v.Name,!v.IsRowMajorProjection),O=View(o.Name,!o.IsRowMajorProjection),
                    QBias=View(q.Bias),KBias=View(k.Bias),VBias=View(v.Bias),OBias=View(o.Bias),Norm=View(norm.Name),NormBias=View(norm.Bias),
                    Heads=cfg.NumHeads,KvHeads=cfg.NumKvHeads,HeadDim=cfg.HeadDim,RotaryDim=checked((int)rotary),
                    NormKind=normKind,Layer=l,Epsilon=cfg.NormEps,RopeBase=cfg.RopeTheta };
            }
            var headReceipts=new AttentionReceipt[attention.Length];var states=new double[checked(attention.Length*words.Length*cfg.HiddenSize)];
            ulong[] rows=sourceRows.Select(index=>checked((ulong)index)).ToArray();
            rc=Attention(attention,(nuint)attention.Length,rows,(nuint)rows.Length,states,(nuint)states.Length,headReceipts,error,(nuint)error.Length);
            if(rc!=0)throw new InvalidDataException(Encoding.UTF8.GetString(error).TrimEnd('\0'));
            Console.WriteLine($"attention_context={string.Join(',',words)} source_indexes={string.Join(',',rows)} heads={headReceipts.Sum(r=>(long)r.Heads)} kv_head_pairs={headReceipts.Sum(r=>(long)r.KvHeads)} head_width={cfg.HeadDim} rotary_dimensions={rotary} input_norm={(normKind==0?"RMSNorm":"LayerNorm")}");
            Console.WriteLine($"attention_projection_error_max={headReceipts.Max(r=>r.ProjectionError):R} attention_head_sum_error_max={headReceipts.Max(r=>r.OutputError):R} causal_prefix_error_max={headReceipts.Max(r=>r.PrefixError):R} position_delta_max={headReceipts.Max(r=>r.PositionDelta):R} value_context_delta_max={headReceipts.Max(r=>r.ContextDelta):R} attention_plus_residual_values={states.Length}");
        }
        snapshot.VerifySourceId();
        Console.WriteLine($"model={Path.GetFileName(Path.GetDirectoryName(Path.GetDirectoryName(path)))} source_snapshot={Convert.ToHexStringLower(snapshot.SourceId.ToBytes())}");
        Console.WriteLine($"canonical_word={word} entity={actual} model_token_index={tokenIndex} tokenizer_provider=PostgreSQL-ordered-vocabulary tensors={headers.Count} layers={layers.Length}");
        Console.WriteLine($"connected_ffn_units={receipts.Sum(r=>(long)r.Units)} decoded_values={receipts.Sum(r=>(long)r.DecodedValues)} max_native_working_array_bytes={receipts.Max(r=>r.ScratchBytes)} relative_error_max={receipts.Max(r=>r.RelativeError):R} absolute_error_max={receipts.Max(r=>r.MaximumError):R}");
        Console.WriteLine($"input=source_embedding_row independent_ffn_results={layers.Length} result_values={results.Length} result_l2={Math.Sqrt(results.Sum(v=>v*v)):R}");
        Console.WriteLine($"binding_seconds={bindingSeconds:F3} total_seconds={watch.Elapsed.TotalSeconds:F3} retained_weight_bytes=0");
    }
    static string Hex(Hash128 id)=>Convert.ToHexStringLower(id.ToBytes());
    static unsafe Hash128[] OrderedIds(byte[] ewkb) {
        fixed(byte* bytes=ewkb) {
            var path=PathOf((IntPtr)bytes,(nuint)ewkb.Length);
            nuint count=PathLength(path);
            if(count==0||count>int.MaxValue)throw new InvalidDataException("invalid stored Native path");
            var result=new Hash128[(int)count];
            if(ExpandPath(path,result,count)!=count)throw new InvalidDataException("stored Native ordinals changed");
            return result;
        }
    }
    static Dictionary<string,Hash128[]> Paths(NpgsqlConnection db,string[] ids) {
        using var command=new NpgsqlCommand("SELECT entity::text,ST_AsEWKB(path) FROM laplace_paths($1::text[]::blake3[])",db);
        command.Parameters.AddWithValue(ids);
        var result=new Dictionary<string,Hash128[]>(StringComparer.Ordinal);
        using var reader=command.ExecuteReader();
        while(reader.Read())result.Add(reader.GetString(0),OrderedIds(reader.GetFieldValue<byte[]>(1)));
        return result;
    }
    static Hash128[] ReadVocabulary(NpgsqlConnection db,string rootId,byte[]? tokenizer) {
        var root=Paths(db,[rootId])[rootId];
        if(root.Length!=4||root[1]!=Hash128.Text("NFC"))throw new InvalidDataException("tokenizer result has no declared source/NFC/raw/decoded layout");
        var paths=Paths(db,[Hex(root[0]),Hex(root[2]),Hex(root[3])]);
        var sourceFile=paths[Hex(root[0])];
        if(sourceFile.Length!=2)throw new InvalidDataException("tokenizer source file has no metadata/content trunk");
        int? sourceIdSpace;
        using(var source=new NpgsqlCommand("SELECT laplace_text($1::text::blake3)",db)) {
            source.Parameters.AddWithValue(Hex(sourceFile[1]));
            string text=(string)(source.ExecuteScalar()??throw new InvalidDataException("tokenizer source content missing"));
            byte[] sourceBytes=Encoding.UTF8.GetBytes(text);
            if(tokenizer is not null&&!sourceBytes.AsSpan().SequenceEqual(tokenizer))throw new InvalidDataException("admitted tokenizer source differs from held package");
            sourceIdSpace=LlamaTokenizerParser.IdSpace(sourceBytes);
        }
        var raw=paths[Hex(root[2])];var decoded=paths[Hex(root[3])];
        if(raw.Length!=decoded.Length||decoded.Length==0||decoded.Length!=sourceIdSpace)throw new InvalidDataException("raw/decoded/source index cardinalities differ");
        Console.WriteLine($"tokenizer_binding_root={rootId} source_file={Hex(root[0])} source_indexes={decoded.Length} source_contract_read_from_postgres=true package_bytes_verified={(tokenizer is not null?"true":"false")}");
        return decoded;
    }
    static int[] SelectRows(NpgsqlConnection db,Hash128[] vocabulary,string[] words,string[] expected) {
        var selected=Enumerable.Repeat(-1,words.Length).ToArray();
        for(int i=0;i<words.Length;i++)selected[i]=Array.IndexOf(vocabulary,Hash128.FromBytes(Convert.FromHexString(expected[i])));
        if(selected.Any(index=>index<0)) {
            // Reduce to declared source pieces and indexed constituent membership
            // before realizing a small candidate set. No per-token database read.
            using var command=new NpgsqlCommand("SELECT p.entity::text,laplace_text(p.entity) FROM physicality p WHERE p.entity=ANY($1::text[]::blake3[]) AND p.path && $2::text[]::blake3[] ORDER BY p.entity",db);
            command.Parameters.AddWithValue(vocabulary.Select(Hex).Distinct().ToArray());
            command.Parameters.AddWithValue(expected);
            using var reader=command.ExecuteReader();
            while(reader.Read()) {
                string pieceRoot=reader.GetString(0);byte[] content=Encoding.UTF8.GetBytes(reader.GetString(1));
                var segments=TextDecomposer.WordSegments(content);
                if(segments.Length!=1)continue;
                for(int i=0;i<words.Length;i++)if(selected[i]<0&&Hex(segments[0].Id)==expected[i]) {
                    selected[i]=Array.IndexOf(vocabulary,Hash128.FromBytes(Convert.FromHexString(pieceRoot)));
                    Console.WriteLine($"constituent_binding={words[i]} entity={expected[i]} source_piece_root={pieceRoot} source_index={selected[i]} byte_offset={segments[0].Offset} binding_provider=PostgreSQL");
                }
            }
        }
        if(selected.Any(index=>index<0))throw new InvalidDataException("admitted source vocabulary does not contain a requested canonical binding");
        return selected;
    }
    static byte[] Read(Stream stream) { using var m=new MemoryStream(); stream.CopyTo(m); return m.ToArray(); }
}
