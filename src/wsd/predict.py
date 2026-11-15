"""Explicit, offline-capable compact WSD model lifecycle."""

import json
import math
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .letters import build_letters
from .model import WSDModernBertForMaskedLM
from .model_surgery import prune_decoder
from .tokenizer import encode_examples


def decode(example, logits, metadata):
    calibration = metadata.get("calibration", {})
    n = len(example.candidates)
    result = dict(
        example_id=example.id,
        selected_id=None,
        reason="no_candidates",
        probabilities={},
        none_probability=1.0,
        model_revision=metadata.get("model_revision", "local"),
        calibration_version=calibration.get("version", "uncalibrated"),
    )
    if not n:
        return result
    temperature = float(calibration.get("temperature", 1.0))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    for key in ("min_probability", "min_margin"):
        threshold = float(calibration.get(key, 0.0))
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"{key} must be finite and within [0,1]")
    values = torch.as_tensor(logits, dtype=torch.float32)[list(range(n)) + [metadata["nota_index"]]]
    probs = torch.softmax(values / temperature, dim=0).tolist()
    if not all(math.isfinite(p) for p in probs):
        raise ValueError("non-finite model probabilities")
    result["probabilities"] = {c.id: p for c, p in zip(example.candidates, probs)}
    result["none_probability"] = probs[-1]
    best = max(range(n + 1), key=lambda i: probs[i])
    ranked = sorted(probs, reverse=True)
    accepted = (
        not calibration.get("abstain_all", False)
        and ranked[0] >= calibration.get("min_probability", 0.0)
        and ranked[0] - ranked[1] >= calibration.get("min_margin", 0.0)
    )
    result["reason"] = ("none_fits" if best == n else "selected") if accepted else "uncertain"
    if accepted and best < n:
        result["selected_id"] = example.candidates[best].id
    return result


def validate_metadata(metadata, tokenizer, max_position_embeddings=None):
    if metadata.get("template_version") != "compact-v1":
        raise ValueError("unsupported prompt template_version")
    letters = metadata.get("letters")
    if not isinstance(letters, list) or len(letters) != 128 or any(not isinstance(x, str) or not x for x in letters):
        raise ValueError("answer mapping needs 128 nonempty strings")
    if len(set(letters)) != 128 or metadata.get("nota_index") != 127:
        raise ValueError("invalid answer mapping")
    token_ids = [tokenizer.encode(" " + letter, add_special_tokens=False) for letter in letters]
    if any(len(ids) != 1 or ids[0] == tokenizer.unk_token_id for ids in token_ids):
        raise ValueError("answer mapping must use known single-token symbols")
    if len({ids[0] for ids in token_ids}) != 128:
        raise ValueError("answer symbols must map to distinct token IDs")
    maximum = metadata.get("max_length")
    if (
        type(maximum) is not int
        or maximum < 1
        or max_position_embeddings is not None
        and maximum > max_position_embeddings
    ):
        raise ValueError("max_length must fit model positional limit")
    if tokenizer.mask_token is None or tokenizer.pad_token is None:
        raise ValueError("tokenizer needs mask and padding tokens")
    for key in ("separator", "none_text"):
        if not isinstance(metadata.get(key), str) or not metadata[key] or tokenizer.mask_token in metadata[key]:
            raise ValueError(f"invalid prompt {key}")
    calibration = metadata.get("calibration", {})
    for key, default, lower, upper in (
        ("temperature", 1.0, 0.0, float("inf")),
        ("min_probability", 0.0, 0.0, 1.0),
        ("min_margin", 0.0, 0.0, 1.0),
    ):
        value = float(calibration.get(key, default))
        if not math.isfinite(value) or not lower <= value <= upper or key == "temperature" and value == 0:
            raise ValueError(f"invalid calibration {key}")
    if "abstain_all" in calibration and type(calibration["abstain_all"]) is not bool:
        raise ValueError("abstain_all must be boolean")


class Predictor:
    def __init__(self, model, tokenizer, metadata, device="cpu"):
        validate_metadata(metadata, tokenizer, model.config.max_position_embeddings)
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.device = device

    @classmethod
    def from_pretrained(cls, path_or_id, revision=None, local_files_only=False, device="cpu"):
        path = Path(path_or_id)
        if not path.is_dir():
            path = Path(
                snapshot_download(
                    str(path_or_id),
                    revision=revision,
                    local_files_only=local_files_only,
                    allow_patterns=["*.json", "*.safetensors", "pytorch_model.bin", "*.txt", "*.model"],
                )
            )
        snapshot_revision = path.name if path.parent.name == "snapshots" else None
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = WSDModernBertForMaskedLM.from_pretrained(path, attn_implementation="sdpa")
        initialized_from_base = model.decoder.out_features != 128
        if initialized_from_base:
            letters = prune_decoder(model, tokenizer)
        else:
            letters = None
        file = path / "model_metadata.json"
        if file.exists():
            metadata = json.loads(file.read_text())
        else:
            sidecar = path / "answer_letters.json"
            saved = json.loads(sidecar.read_text()) if sidecar.exists() else None
            # This published upstream format predates self-describing metadata.
            # Compatibility is explicit to its identity, never guessed from hidden size.
            known_upstream = {"sign/Ettin-150m-WSD", "sign/Ettin-1B-WSD", "sign/ModernBERT-Large-Instruct-WSD"}
            legacy = str(path_or_id) in known_upstream or path.parent.parent.name in {
                "models--" + name.replace("/", "--") for name in known_upstream
            }
            if not legacy and not initialized_from_base:
                raise ValueError("Compact checkpoint needs model_metadata.json; prompt format cannot be inferred")
            mapping = saved["letters"] if saved else list((letters or build_letters(tokenizer)).letters)
            metadata = dict(
                letters=mapping,
                nota_index=127,
                template_version="compact-v1",
                separator="[unused0]" if legacy else (tokenizer.sep_token or tokenizer.eos_token or "\n"),
                none_text="none of the above",
                max_length=2048,
                model_revision=revision or (path.name if path.parent.name == "snapshots" else "local"),
                calibration=dict(temperature=1.0, min_probability=0.0, min_margin=0.0, version="uncalibrated"),
            )
        if len(metadata["letters"]) != 128 or metadata["nota_index"] != 127:
            raise ValueError("invalid answer mapping")
        if len(set(metadata["letters"])) != 128:
            raise ValueError("duplicate answer letters")
        calibration = path / "calibration.json"
        if calibration.exists():
            metadata["calibration"] = json.loads(calibration.read_text())
        if snapshot_revision:
            metadata["model_revision"] = snapshot_revision
        return cls(model, tokenizer, metadata, device)

    def logits(self, examples):
        if not examples:
            return torch.empty(0, 128)
        batch = {k: v.to(self.device) for k, v in encode_examples(examples, self.tokenizer, self.metadata).items()}
        with torch.inference_mode():
            return self.model(**batch).logits.float().cpu()

    def predict(self, examples):
        for example in examples:
            example.validate()
        valid = [e for e in examples if e.candidates]
        rows = iter(self.logits(valid))
        return [decode(e, next(rows) if e.candidates else torch.zeros(128), self.metadata) for e in examples]

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        (path / "model_metadata.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (path / "calibration.json").write_text(
            json.dumps(self.metadata.get("calibration", {}), indent=2) + "\n", encoding="utf-8"
        )
