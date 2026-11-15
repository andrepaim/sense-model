# Export feasibility receipt — 2026-11-15

This is a mechanical export check, not a quality or Windows latency benchmark.

Source checkpoint: `sign/Ettin-150m-WSD`, cached revision `8751b577199d1bb95b74fa2457da7065d57100ae`. Exported on Linux CPU with PyTorch 2.15.0. Inputs: six authored smoke fixtures (`data/fixtures/smoke.jsonl`), including Portuguese strings used only to check Unicode/tokenization. The English checkpoint has no validated Portuguese support.

Command:

```sh
wsd export --model /path/to/checkpoint --fixtures data/fixtures/smoke.jsonl --output artifacts/upstream-english --quantize
cd examples/node
WSD_ARTIFACT=../../artifacts/upstream-english npm test
```

Float ONNX max absolute logit error: `9.54e-6` across the batch of six examples and each example independently (32–54 input tokens). INT8 chosen-ID agreement on those six fixtures: 100%; labelled fixture accuracy loss: 0.0. Maximum INT8 raw logit drift: 2.203, demonstrating why broad evaluation and recalibration must precede release claims. FP32 ONNX size is approximately 572 MiB; dynamic MatMul INT8 ONNX approximately 255 MiB. Tokenizer assets are additional. This export quantizes MatMul weights and does not promise all-parameter INT8 storage.

Additional dynamic-shape checks used real checkpoint inputs of 37, 136, 164, 292 and 548 tokens, crossing the 128-token local attention window. Max singleton logit error: `1.03e-5`; max mixed padded-batch error: `6.20e-6`. Local attention behavior was preserved without changing model attention implementations.

Node actual tokenizer + ONNX Runtime tests passed all three checks, including probabilities and decisions against the exported Python reference at `1e-4` tolerance. Dependencies are locked; the direct `onnxruntime-node` version matches Transformers.js's native runtime to avoid duplicate library ABI conflicts. `npm audit` reported no known vulnerabilities at installation.

Full-precision and quantized ONNX files plus detailed tensor fixtures are local ignored artifacts under `artifacts/upstream-english/`. They have not been uploaded to Hugging Face. Frozen real-text evaluation, calibration and ordinary Windows laptop CPU resource measurements remain necessary before a product release.
