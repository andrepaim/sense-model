import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import ModernBertConfig, PreTrainedTokenizerFast

from wsd.model import WSDModernBertForMaskedLM
from wsd.predict import Predictor, decode
from wsd.schema import Candidate, Example, Target, utf16_to_codepoint
from wsd.tokenizer import encode_examples


def tiny_predictor():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[unused0]": 3, "*": 4}
    vocab.update({chr(0x400 + i): 5 + i for i in range(128)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", mask_token="[MASK]")
    config = ModernBertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=256,
        local_attention=32,
        global_attn_every_n_layers=1,
        answer_vocab_size=128,
        tie_word_embeddings=False,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=1,
    )
    model = WSDModernBertForMaskedLM(config)
    meta = dict(
        letters=[chr(0x400 + i) for i in range(128)],
        nota_index=127,
        template_version="compact-v1",
        separator="[unused0]",
        none_text="none of the above",
        max_length=256,
    )
    return Predictor(model, tokenizer, meta)


def example():
    return Example(
        "bank", "bank bank", Target(5, 9, "bank"), [Candidate("a", "seat"), Candidate("b", "money")], gold_ids=["a"]
    )


def test_roundtrip(tmp_path):
    p = tiny_predictor()
    x = example()
    logits = p.logits([x])
    assert logits.shape == (1, 128)
    p.save_pretrained(tmp_path)
    q = Predictor.from_pretrained(tmp_path, local_files_only=True)
    torch.testing.assert_close(logits, q.logits([x]))
    assert p.predict([x]) == q.predict([x])


def test_span_and_limits():
    x = example()
    assert "*bank*" in __import__("wsd.tokenizer", fromlist=["format_prompt"]).format_prompt(
        x, tiny_predictor().tokenizer, tiny_predictor().metadata
    )
    x.target = Target(4, 8, "bank")
    with pytest.raises(ValueError):
        x.validate()
    assert utf16_to_codepoint("😀banco", 2) == 1
    with pytest.raises(ValueError):
        utf16_to_codepoint("😀banco", 1)


def test_none_uncertain_and_empty():
    x = example()
    meta = dict(nota_index=127, calibration={"min_probability": 0.9})
    logits = torch.zeros(128)
    assert decode(x, logits, meta)["reason"] == "uncertain"
    logits[127] = 10
    assert decode(x, logits, meta)["reason"] == "none_fits"
    x.candidates = []
    x.gold_ids = []
    assert tiny_predictor().predict([x])[0]["reason"] == "no_candidates"


def test_literal_mask_and_length():
    p = tiny_predictor()
    x = example()
    x.candidates[0].definition = "[MASK]"
    with pytest.raises(ValueError):
        p.predict([x])
    x = example()
    p.metadata["max_length"] = 2
    with pytest.raises(ValueError):
        encode_examples([x], p.tokenizer, p.metadata)
