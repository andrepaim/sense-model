Implementation contract (shared among development tasks)
- package src/wsd. Python >=3.11. dataclasses in schema.py:
  Candidate(id: str, definition: str)
  Target(start: int, end: int, text: str)
  Example(id: str, text: str, target: Target, candidates: list[Candidate], language: str='en', definition_language: str='en', gold_ids: list[str] | None=None, provenance: dict=field(default_factory=dict)). gold_ids=[] means none fits; None means unlabeled.
  Example.from_dict(dict), .to_dict(), .validate(). offsets code points exclusive end.
- predict.py: Predictor.from_pretrained(path_or_id, revision=None, local_files_only=False, device='cpu'); .predict(examples: list[Example]) -> list[dict]; .logits(examples)-> torch.Tensor [B,128]; .save_pretrained(path). attributes model,tokenizer,metadata (dict).
- tokenizer.py: encode_examples(examples, tokenizer, metadata) -> dict[str,Tensor] input_ids,attention_mask,prediction_positions. metadata contains letters list, nota_index=127, template_version='compact-v1', separator='[unused0]', none_text='none of the above', max_length=2048. full prompts exceeding max_length raise; literals '*' or mask token in text/definitions rejected for legacy safety.
- data.py: read_jsonl(path)->list[Example]; write_jsonl(path,examples); validate_splits(paths: dict[str,path])->dict (leakage and fingerprints); prepare CLI independent.
- train.py main(argv=None), evaluate.py main(argv=None), export.py main(argv=None), publish.py main(argv=None). cli.py lazily dispatches subcommands train,evaluate,export,publish,prepare,predict,teacher.
- Predict response: example_id, selected_id, reason (selected/none_fits/uncertain/no_candidates), probabilities dict keyed candidate id, none_probability, model_revision, calibration_version. metadata calibration={temperature:1.0,min_probability:0.0,min_margin:0.0,version:'uncalibrated'}.
- all file writes UTF-8; explicit external publication only. No implicit data/model downloads except explicitly loading a named HF model.
