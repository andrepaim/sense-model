# sense-model

This is a standalone candidate-definition selection model library. Keep inference
independent of dictionary lookup, linguistic pipelines, HTTP services and pdfpal.
Run all commands from this repository, because the sibling upstream repository
also has a package named wsd.

Use .venv/bin/python -m pytest and uv lock --check. Keep tests offline using tiny
ModernBERT fixtures. Real checkpoint checks and ONNX parity are release checks.
Never describe synthetic smoke fixtures as benchmark results. Publishing is an
explicit wsd publish --upload action; training never uploads automatically.
Preserve NOTICE and upstream MIT attribution when adapting source.
