# Portuguese training infrastructure smoke — 2026-11-15

A real mmBERT-small checkpoint was initialized with a compact WSD head, trained for three CPU steps, saved and reloaded. A separate checkpoint-resume run restored optimizer/scheduler state and reached global step 2. Inputs were one authored synthetic training context and one different synthetic development context. This validates execution only; it establishes no Portuguese language quality or improvement.

```json
{
  "checkpoint": "jhu-clsp/mmBERT-small",
  "revision": "abc32620dd4f6ab06f5fbe905dc25f310618e09f",
  "device": "cpu",
  "threads": 4,
  "fixture_annotation": "authored synthetic; smoke only",
  "train_count": 1,
  "dev_count": 1,
  "optimization_steps": 3,
  "changed_tensors": 0,
  "total_tensors": 138,
  "resume_global_step": 2,
  "development_metrics": {
    "eval_loss": 1.1729364395141602,
    "eval_accuracy": 0.0,
    "eval_runtime": 0.0228,
    "eval_samples_per_second": 43.933,
    "eval_steps_per_second": 43.933,
    "epoch": 3.0
  },
  "limitation": "Infrastructure smoke only. No Portuguese capability, quality improvement, standard evaluation, or release readiness is established.",
  "changed_tensors_at_last_checkpoint": 138,
  "final_selection": "First checkpoint selected because all synthetic development accuracies tied; final best model unchanged."
}
```

Reproduction commands (with local fixture paths):

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 wsd train --model artifacts/portuguese-initialized --train artifacts/portuguese-smoke-data/train.jsonl --dev artifacts/portuguese-smoke-data/dev.jsonl --output artifacts/portuguese-update-smoke --max-steps 3 --save-steps 1 --patience 5 --batch-size 1 --max-length 256 --local-files-only
```

The fixture data and large model artifacts remain local under the ignored artifacts directory. Unit tests independently reproduce training/save/reload/resume using a tiny ModernBERT model without downloading weights.
