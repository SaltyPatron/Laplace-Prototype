# Model construction consumer checks

The monorepo owns the source selection, typed path compilation and native tensor construction.
These tools inspect and execute its conventional model exports. They do not load a pretrained
checkpoint or ask an external model to write responses.

The native full-recipe exporter now constructs the output head in-process against
its executed F32 attention/FFN states. It does not require the older external
readout calibrator below.

- `laplace-verify-native-states MODEL MODEL.states.bin [MODEL.targets.bin]` loads
  the standalone model in CPU llama.cpp and compares every recorded canonical and
  ordered-prefix state. With targets supplied it also measures actual runtime
  logits, candidate membership, and total probability outside each candidate set.
- `laplace-safetensors-consumer SAFETENSORS_DIR REFERENCE.gguf OUTPUT.gguf`
  consumes every exported HF tensor, verifies names/shapes/finite values, packs
  expert tensors and writes a CPU-consumable container. The reference supplies
  tokenizer/architecture metadata; its weights are not used. Run the state
  verifier on the result against the SafeTensors export's own state receipt.

Native full-recipe exports record `MODEL.basis.json`, `MODEL.operators.json`,
`MODEL.tensors.json`, `MODEL.calculations.json`, and provenance/source-attribution
receipts. SafeTensors directories contain copies of these JSON receipts. A
requested output probability floor is a target policy; inspect measured outside
mass and candidate membership rather than assuming projection met that target.

- `inspect-gguf.mjs MODEL REPORT.json` validates the container and canonical BLAKE3 map,
  checks all F32 tensors for nonfinite values, and records tensor and head-slice hashes.
  A zero-layer lookup additionally gets exact direct-readout diagnostics.
- `run-prompts.mjs MODEL CPU_COMPLETION OUTPUT_DIR` records all eight June prompts,
  their bare forms and case controls, including exact arguments and full process diagnostics.
- `laplace-readout-calibrate MODEL PROGRAM.bin` measures the final normalized hidden state
  of `[BOS, token]` for **every** exported vocabulary slot through all generated layers.
  The monorepo's native SGELSD operation constructs the output weights from those states
  and the selected path targets. Only `output.weight` is replaced. The result is standalone.
- `laplace-readout-calibrate MODEL PROGRAM.bin --evaluate` measures the runtime's actual
  output logits for the complete vocabulary without changing weights. It reports active-path
  and empty-path populations separately. This checks the specified construction domain;
  it does not score general conversation or unseen contexts.
- `ablate-gguf.mjs MODEL REPORT.json NEW_MODEL attention|ffn|both` creates a new copy with
  selected attention-output and/or FFN-down matrices zeroed. The output head remains fixed.
  Evaluate that copy with the same program to measure the contribution of the generated body.

Run the JavaScript tools with Node. Build the CPU consumer with:

```sh
cmake -S models -B /tmp/laplace-model-consumer \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_SOURCE_DIR=/path/to/llama.cpp \
  -DLLAMA_LIBRARY_DIR=/path/to/llama.cpp/build-cpu/bin \
  -DLAPLACE_SOURCE_DIR=/path/to/Laplace \
  -DLAPLACE_SYNTHESIS_DIR=/path/to/built/synthesis
cmake --build /tmp/laplace-model-consumer --parallel 6
```

Use a CPU-only llama.cpp build. This adapter links its CPU libraries directly; it does
not initialize a GPU backend. The synthesis library must include the current native
`laplace_readout_fit` API. Set a recipe's `runtime_calibrator` to the built executable
for construction to run inside `synthesize substrate` and update its tensor receipts.
The binary program beside the export contains explicit input/output realization maps
and weighted COO path endpoints. Canonical token identifiers stay distinct.

The path input representation uses the selected transfer's output columns and S³
coordinates, conditions that column space by native SVD, and adds a constant-norm slack
coordinate and bias. The Q/K score and V/O value equations are distinct:
`R S Rᵀ ≈ A` and `R V ≈ A R`. Repeated heads receive separate mode ranges distributed
across the selected heads. Projection errors and the final runtime readout error are
measurements, not guarantees of contextual behavior. Provenance is recorded in
`MODEL.provenance.json`, tensor hashes in `MODEL.tensors.json`, and the runtime solve
in `MODEL.calibration.json` with full diagnostics in `MODEL.calibration.log`.

## Tokenizer source admission before numeric extraction

`verify-tokenizer-source.mjs` runs the existing Native Engine on the selected
packages' `tokenizer.json` files, reconstructs them from PostgreSQL through the
existing MCP `file_outline` operation, compares the complete bytes and SHA-256,
and repeats admission to check that both recorded file trunks are skipped with
zero new entity/path rows. It writes a bounded receipt and the Engine's logs to
the selected external destination. Connection credentials are not recorded.

```sh
LAPLACE_CONNINFO='host=/tmp user=laplace dbname=laplace' \
LAPLACE_RECIPES=/hosting/laplace/modular/current/share/recipes \
node models/verify-tokenizer-source.mjs \
  /hosting/laplace/modular/current/bin/laplace \
  /hosting/laplace/modular/current/bin/lpm \
  /repos/work/tier0/tier0.bin \
  /repos/archive/laplace/tokenizer-source-check.json \
  /path/to/TinyLlama/snapshot /path/to/phi-2/snapshot
```

This is the source-structure admission step. The existing JSON grammar retains
vocabulary members/indexes, added-token records, merge order and decoder syntax
in shared canonical compositions under the file's metadata/content trunk.
The check implements no tokenizer decoder or model arithmetic. Decoded-piece
bindings, their applicable source conditions, model-trunk integration and learned
effects are subsequent handoffs; this receipt does not establish those handoffs.
The vocabulary counts describe source IDs, not distinct canonical text identities.

The real TinyLlama and Phi-2 tokenizer sources were admitted together on
2026-10-29 in 39.9 seconds: 1,842,767 and 2,114,924 bytes, with 32,000 and 50,295
distinct source token IDs respectively. Phi includes 38 added-only IDs. Independent
MCP reconstruction matched both files byte for byte. The immediate Engine replay
skipped both inputs in 1.3 seconds with zero new entity/path rows. These timings
are for tokenizer source structure, not complete model ETL. The original admission,
readback and replay receipts are under `/repos/archive/laplace/` on hart-server.

## Decoded tokenizer admission and overlap

The provider admits its complete decoded vocabulary with `--admit`. It verifies
the original tokenizer file's persisted content against the input, then uses the
existing Core `canonical_stage` API for composition, top-down presence descent
and paired binary COPY. PostgreSQL's existing `laplace_merge_stage` performs
publication. Caller-owned UNLOGGED inputs use the normal `stage` schema and are
dropped before commit; failures roll back their creation. No per-token SQL
transport or private durable schema is introduced.

The result has four ordered constituents: the original tokenizer file trunk,
the canonical `NFC` policy, the raw vocabulary and the decoded vocabulary. Each
vocabulary's logical ordinal is its source index, including repeats. Leading
spaces and case remain in decoded content; the source tree retains decoder,
merge, control and boundary conditions. Incomplete UTF-8 pieces have
`[Byte, ordered canonical decimal byte values]` descriptors, using ordinary
Unicode content instead of another Tier-0 alphabet or a replacement character.
This result is a tokenizer branch for the later complete model trunk. Recording
it registers no separate model witness and creates no learned semantic claims.

```sh
LAPLACE_TIER0=/repos/work/tier0/tier0.bin \
LAPLACE_CONNINFO='host=/tmp user=laplace dbname=laplace' \
/repos/build/Laplace-Prototype/models/tokenizer-overlap/release/laplace-tokenizer-overlap \
  --admit /path/to/snapshot/tokenizer.json PERSISTED_TOKENIZER_FILE_ID \
  /repos/archive/laplace/tokenizer-bindings.json
```

Real admissions on 2026-10-29 verified every raw and decoded ordinal before commit:

| Model | Source indexes | New entity/path pairs | Admission seconds | Replay new pairs |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama | 32,000 | 16,316 | 9.516 | 0 |
| Phi-2 | 50,295 | 22,902 | 12.207 | 0 |

Each initial admission used 19 SQL commands and 4 Native presence rounds; each
replay used 8 commands and one root presence round. PostgreSQL confirms 18,222
shared entries between the decoded manifests (31,898 and 50,295 distinct IDs,
including byte descriptors). Result roots are
`3a85230c76cb66b7d1e1995951ffc68d` and
`b41f653da86b3483520843eff1287d96`. Initial and replay receipts are
`/repos/archive/laplace/{tiny,phi}-tokenizer-bindings[-replay]-20261029.json`.

Without `--admit`, `tokenizer-overlap` compares both complete source vocabularies, including added
IDs. Hugging Face's pinned Rust tokenizer executes the declared decoders. A strict
byte check matches every complete UTF-8 piece against that decoder and keeps
incomplete byte fragments separate instead of merging replacement characters.
The receiver initializes the ordinary Native codepoint perf-cache, uses Native
canonical content IDs and UAX segmentation, and applies NFC with case preserved.
The whitespace-neutral comparison is a comparison view; exact decoded pieces
and source indexes remain available independently. Special-role pieces and empty
whitespace-only results are excluded from the neutral and UAX comparisons.

```sh
RUSTFLAGS='-Lnative=/opt/laplace/app/cli -Clink-arg=-Wl,-rpath,/opt/laplace/app/cli' \
cargo build --release --locked --jobs 2 \
  --manifest-path models/tokenizer-overlap/Cargo.toml \
  --target-dir /repos/build/Laplace-Prototype/models/tokenizer-overlap
LAPLACE_TIER0=/repos/work/tier0/tier0.bin \
/repos/build/Laplace-Prototype/models/tokenizer-overlap/release/laplace-tokenizer-overlap \
  /path/to/TinyLlama/snapshot/tokenizer.json \
  /path/to/phi-2/snapshot/tokenizer.json \
  /repos/archive/laplace/tokenizer-overlap.json
```

The observed decoder contracts currently qualified here are TinyLlama's
Replace/ByteFallback/Fuse/Strip sequence and Phi's ByteLevel decoder. Comparison
mode is read-only; it does not replace production ingestion or
enumerate the text that could be assembled across arbitrary token sequences.

Real-source runs on 2026-10-29, 1.27–1.50 seconds including Native cache initialization:

| Comparison | TinyLlama distinct | Phi-2 distinct | Shared |
| --- | ---: | ---: | ---: |
| Original source spellings | 32,000 | 50,295 | 7,263 |
| Exact decoded bytes, including byte fragments | 31,904 | 50,295 | 18,222 |
| NFC text, surrounding whitespace removed, excluding special/empty pieces | 26,457 | 41,400 | 14,745 |
| Native UAX segments containing letters/numbers, including subword fragments | 25,228 | 40,626 | 14,238 |

The neutral text union is 53,112 distinct pieces. Shared pieces are 55.73% of
TinyLlama's distinct neutral pieces and 35.62% of Phi's. Incomplete/invalid UTF-8
sequences are tracked separately: 128 distinct in TinyLlama, 344 in Phi, 128
shared. These counts describe represented content and fragments, not learned
semantic agreement or the final number of admitted DAG nodes. Receipt:
`/repos/archive/laplace/tiny-phi-tokenizer-overlap-recorded-20261029.json`.

## Connected source-component calculations

`model-components` exercises the ingestion handoff from admitted PostgreSQL
tokenizer bindings and the production SafeTensors parser/shape recognizer into
native component calculations.
The C# adapter holds the source snapshot, passes its open file handles and recognized
tensor ranges to C++, and resolves the chosen word's identity against PostgreSQL.
It reads the qualified ordered vocabulary from PostgreSQL and verifies the source
against the held package. The spelling-heuristic parsing fallback is removed from
this consumer. `LlamaTokenizerParser.IdSpace` checks source index cardinality only;
tensor roles remain owned by the selected production runtime.
Before running the FFN calculation it checks every source tensor against the
recognized roles and biases, and reports attention, FFN, normalization and
vocabulary bindings separately. Unbound tensors, ambiguous slots or conflicting
anatomy stop the run. Tensor-binding coverage and calculated FFN effects have
distinct receipt fields; sequence/position, masking and residual wiring also
belong to the complete model execution contract.

Supply comma-separated canonical operands, such as `king,queen`, to include attention.
Native `lp_path_expand` reads stored ordinals, including repeated IDs. Indexed
membership reduces to candidate pieces within that admitted vocabulary before
Native UAX `TextDecomposer.WordSegments` resolves constituents. For example, `queen` is a
constituent of the space-prefixed piece; that piece keeps its own content root,
source index and decoded-content byte offset. This uses containment instead of discarding the space
or requiring the entire tokenizer piece to equal the requested word.

The Native attention calculation applies the source input norm (RMSNorm or
LayerNorm), Q/K/V projections and biases, full or partial RoPE, scaled causal
softmax, shared-KV value selection, the per-head O-column contributions and output
bias, and adds the residual input. Q/K/V row calculations are compared with the
existing bulk Native projection; summed O contributions are compared with a whole
projection of concatenated attended values. Prefix invariance and controlled
position/value changes exercise the context dependencies. Current supported source
options are the unscaled causal RoPE layouts without separate Q/K normalization
used by the selected TinyLlama and Phi-2 packages. The source-bound formula follows
the [Phi reference implementation](https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/phi/modeling_phi.py).
Numerical comparisons use F64 working arithmetic over the source F16/BF16 values.

Each attention layer is checked independently on the selected source embedding
rows. Its returned `input + attention` state is an intermediate result available
to further Native calculations. The FFN calculation remains a separate component
check. The source architecture determines how these branches, subsequent norms
and readout connect; these component checks do not execute that complete model
graph or publish its semantic evidence.

The native call visits every FFN unit in every recognized layer: the corresponding
up/gate rows and down column, with the declared activation and optional biases.
Each layer is calculated independently using the chosen token's source embedding
as its numerical input. Component effects are summed and compared with the existing
`ffn_write_vectors_ex_d` complete FFN kernel. The API returns each layer's result
vector in memory for subsequent calculations; arithmetic and source weights remain
transient. The receipt reports unit counts, numerical comparison error, working-array
bytes and time. This check exercises connected-component decoding and calculation.

Build on Linux with the installed production managed/native runtime and an external
build directory (the C++ compiler must support F16C):

```sh
cmake -S models -B /path/to/component-build \
  -DLAPLACE_COMPONENT_ONLY=ON -DCMAKE_BUILD_TYPE=Release \
  -DLAPLACE_SOURCE_DIR=/path/to/Laplace -DLAPLACE_RUNTIME_DIR=/opt/laplace/lib
cmake --build /path/to/component-build --target laplace-model-components
dotnet build models/model-components.csproj -c Release \
  -p:LaplaceManagedDir=/opt/laplace/app/cli \
  -p:ComponentBuildDir=/path/to/component-build
dotnet /path/to/component-build/managed/model-components.dll \
  /path/to/tier0.bin 'Host=/tmp;Username=laplace;Database=laplace' king,queen \
  --binding-root 3a85230c76cb66b7d1e1995951ffc68d /path/to/TinyLlama/snapshot
```

Use Phi's own result root and package for its component check. With `--bindings
ROOT` instead of `--binding-root ROOT PACKAGE`, the reader resolves source indexes
wholly from PostgreSQL, with no model source files opened. These operands select
concrete source rows for component checks; complete source-sequence encoding has
its own boundary/context contract. Real `king,queen` checks selected Tiny indexes
9292/26624 and Phi3364/16599 from the admitted manifests; the existing numerical
reference results were unchanged.

On hart-server, 2026-10-29, the real TinyLlama and Phi-2 packages supplied 123,904
and 327,680 connected FFN units respectively. Both tokenizers bound `king` to
`b25c2d2149dfbb3889de432669062b54`, matching PostgreSQL, at source indexes 9292
and 3364. Component sums matched the complete native operators within
5.27e-16 and 5.18e-15 maximum relative error over their layers. These measurements
use one numerical input per independent layer, with all units participating.

The two-entry attention checks ran in both `king,queen` and `queen,king` order.
They covered 704 query-head routes sharing 88 KV pairs in TinyLlama and 1,024
routes/KV pairs in Phi. Source head widths were 64/80, with 64/32 rotary
coordinates respectively. Maximum relative Q/K/V projection discrepancy was
2.19e-15, and head-sum discrepancy 1.30e-15. The causal prefix remained invariant
within 1.51e-15 relative numerical discrepancy. Changing the positional gap or
the final V entry changed the calculated output for these inputs. These are
operator-dependency checks, not semantic similarity or confidence scores.
