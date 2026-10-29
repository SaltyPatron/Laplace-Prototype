using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
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
static class Program {
    [DllImport("laplace-model-components",EntryPoint="lp_model_ffn_components")]
    static extern int Components([In] FfnBinding[] layers,nuint count,ulong row,
        [Out] double[] results,nuint resultCount,[Out] ComponentReceipt[] receipts,
        [Out] byte[] error,nuint errorBytes);

    static int Main(string[] args) {
        try {
            if(args.Length<4) throw new ArgumentException("usage: model-components TIER0 CONNINFO WORD PACKAGE...");
            if(!OperatingSystem.IsLinux()) throw new PlatformNotSupportedException("the held-file view adapter currently uses Linux file descriptors");
            if(Marshal.SizeOf<TensorView>()!=40 || Marshal.SizeOf<FfnBinding>()!=288 || Marshal.SizeOf<ComponentReceipt>()!=40)
                throw new InvalidOperationException("native component ABI layout mismatch");
            CodepointPerfcache.Load(args[0]);
            using var db=new NpgsqlConnection(args[1]); db.Open();
            using var query=new NpgsqlCommand("SELECT laplace_text_id($1)::text",db);
            query.Parameters.AddWithValue(args[2]);
            string expected=(string)query.ExecuteScalar()!;
            foreach(string package in args.Skip(3)) Run(package,args[2],expected);
            return 0;
        } catch(Exception e) { Console.Error.WriteLine(e); return 1; }
    }

    static void Run(string directory,string word,string expected) {
        var watch=Stopwatch.StartNew();
        string path=SafetensorSnapshotWitness.ResolveCompleteDir(directory)
            ?? throw new InvalidDataException("complete model package required");
        using var snapshot=SourceEntityIdConventions.OpenModelContentSnapshot(path)
            ?? throw new InvalidDataException("model source snapshot unavailable");
        var headers=SafetensorsContainerParser.ParseModel(snapshot);
        byte[] tokenizer=snapshot.Read(Path.Combine(path,"tokenizer.json"),Read);
        var config=ModelConfigReader.Read(snapshot.Read(Path.Combine(path,"config.json"),Read));
        var manifest=ModelManifest.Recognize(headers,config,LlamaTokenizerParser.IdSpace(tokenizer),path);
        var tokens=LlamaTokenizerParser.Parse(tokenizer);
        byte[] wanted=Encoding.UTF8.GetBytes(word);
        var token=tokens.FirstOrDefault(t=>t.CanonicalBytes.AsSpan().SequenceEqual(wanted))
            ?? throw new InvalidDataException("canonical word absent from source vocabulary");
        string actual=Convert.ToHexStringLower(token.EntityId.ToBytes());
        if(actual!=expected) throw new InvalidDataException($"canonical binding disagrees with PostgreSQL: {actual} != {expected}");
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
        int rc=Components(layers,(nuint)layers.Length,checked((ulong)token.TokenId),results,(nuint)results.Length,receipts,error,(nuint)error.Length);
        if(rc!=0) throw new InvalidDataException(Encoding.UTF8.GetString(error).TrimEnd('\0'));
        snapshot.VerifySourceId();
        Console.WriteLine($"model={Path.GetFileName(Path.GetDirectoryName(Path.GetDirectoryName(path)))} source={Convert.ToHexStringLower(snapshot.SourceId.ToBytes())}");
        Console.WriteLine($"canonical_word={word} entity={actual} model_token_index={token.TokenId} tokenizer_provider=LlamaTokenizerParser tensors={headers.Count} layers={layers.Length}");
        Console.WriteLine($"connected_ffn_units={receipts.Sum(r=>(long)r.Units)} decoded_values={receipts.Sum(r=>(long)r.DecodedValues)} max_native_working_array_bytes={receipts.Max(r=>r.ScratchBytes)} relative_error_max={receipts.Max(r=>r.RelativeError):R} absolute_error_max={receipts.Max(r=>r.MaximumError):R}");
        Console.WriteLine($"input=source_embedding_row independent_ffn_results={layers.Length} result_values={results.Length} result_l2={Math.Sqrt(results.Sum(v=>v*v)):R}");
        Console.WriteLine($"binding_seconds={bindingSeconds:F3} total_seconds={watch.Elapsed.TotalSeconds:F3} retained_weight_bytes=0");
    }
    static byte[] Read(Stream stream) { using var m=new MemoryStream(); stream.CopyTo(m); return m.ToArray(); }
}
