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
