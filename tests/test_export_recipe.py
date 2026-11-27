"""The quantization recipe of training round 1 in wsd export: activation outliers choose feed-forward output
MatMuls to keep in float, per-channel INT8 weights for the rest, a float16 token embedding with a Cast, and the INT8
graph checked against the float graph on development prompts."""

import json

import pytest
from test_core import tiny_predictor

from wsd.data import write_jsonl
from wsd.schema import Candidate, Example, Target

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from wsd.export import (  # noqa: E402
    activation_outliers,
    export_model,
    first_prompts,
    main,
    outlier_layers,
    outlier_nodes,
)

EMBEDDING = "model.model.embeddings.tok_embeddings.weight"


def fixtures():
    rows = []
    for i, words in enumerate([2, 5, 3, 7]):
        text = " ".join(["bank"] * words)
        rows.append(
            Example(
                f"x{i}", text, Target(0, 4, "bank"), [Candidate("a", "seat"), Candidate("b", "money")], gold_ids=["a"]
            )
        )
    return rows


def test_outlier_rule_picks_feed_forward_outputs_above_the_ratio():
    rows = [
        {"name": "model.layers.0.mlp.Wo", "outlier_ratio": 42.9},
        {"name": "model.layers.1.mlp.Wo", "outlier_ratio": 13.5},
        {"name": "model.layers.2.mlp.Wi", "outlier_ratio": 99.0},
        {"name": "model.layers.10.mlp.Wo", "outlier_ratio": 15.1},
    ]
    assert outlier_layers(rows, 15.0) == [0, 10]
    assert outlier_nodes([0, 10]) == ["/model/model/layers.0/mlp/Wo/MatMul", "/model/model/layers.10/mlp/Wo/MatMul"]


def test_activation_statistics_cover_every_linear_layer():
    p = tiny_predictor()
    rows = activation_outliers(p, fixtures(), count=3)
    names = {r["name"] for r in rows}
    assert "model.layers.0.mlp.Wo" in names and "decoder" in names
    assert all(r["outlier_ratio"] >= 1 and r["act_max_max"] >= r["act_max_mean"] > 0 for r in rows)


def test_first_prompts_are_distinct_and_sorted_by_hash():
    p = tiny_predictor()
    xs = fixtures()
    xs.append(Example("same", xs[0].text, xs[0].target, xs[0].candidates, gold_ids=["b"]))
    chosen = first_prompts(xs, p.tokenizer, p.metadata, 3)
    from wsd.scoring import prompt_sha256

    hashes = [prompt_sha256(x, p.tokenizer, p.metadata) for x in chosen]
    assert len(chosen) == 3 and hashes == sorted(hashes) and len(set(hashes)) == 3


def test_recipe_keeps_named_matmuls_in_float_and_stores_the_embedding_in_float16(tmp_path):
    p = tiny_predictor()
    keep = ["/model/model/layers.0/mlp/Wo/MatMul"]
    report = export_model(
        p,
        fixtures(),
        tmp_path / "out",
        quantize=True,
        per_channel=True,
        exclude_nodes=keep,
        float16_embedding=True,
        max_accuracy_loss=1,
    )
    graph = onnx.load(str(tmp_path / "out" / "onnx" / "model.int8.onnx"))
    nodes = {n.name: n for n in graph.graph.node}
    assert nodes[keep[0]].op_type == "MatMul"
    assert any(n.op_type == "MatMulInteger" for n in graph.graph.node)
    assert graph.graph.node[0].op_type == "Cast" and graph.graph.node[0].output == [EMBEDDING]
    stored = {i.name: i.data_type for i in graph.graph.initializer}
    assert stored[EMBEDDING + "_fp16"] == onnx.TensorProto.FLOAT16 and EMBEDDING not in stored
    assert not (tmp_path / "out" / "onnx" / "model.pc-int8.onnx").exists()
    assert report["int8"]["per_channel"] is True and report["int8"]["excluded_nodes"] == keep
    with pytest.raises(ValueError, match="not in the graph"):
        export_model(p, fixtures(), tmp_path / "bad", quantize=True, exclude_nodes=["/no/such/MatMul"])


def test_export_command_measures_outliers_and_checks_int8_against_float(tmp_path):
    p = tiny_predictor()
    p.save_pretrained(tmp_path / "model")
    write_jsonl(tmp_path / "fixtures.jsonl", fixtures())
    write_jsonl(tmp_path / "dev.jsonl", fixtures()[1:])
    main(
        [
            "--model",
            str(tmp_path / "model"),
            "--local-files-only",
            "--fixtures",
            str(tmp_path / "fixtures.jsonl"),
            "--output",
            str(tmp_path / "out"),
            "--quantize",
            "--per-channel",
            "--float16-embedding",
            "--outlier-examples",
            str(tmp_path / "dev.jsonl"),
            "--outlier-count",
            "2",
            "--outlier-ratio",
            "1.0",
            "--check-examples",
            str(tmp_path / "dev.jsonl"),
            "--check-count",
            "3",
            "--max-accuracy-loss",
            "1",
        ]
    )
    out = tmp_path / "out"
    recipe = json.loads((out / "recipe.json").read_text())
    assert recipe["outlier_rule"] == "mlp.Wo input max/p99.9 > 1.0"
    assert recipe["excluded_nodes"] == outlier_nodes(recipe["outlier_layers"])
    assert recipe["quantize_dynamic"] == {
        "weight_type": "QInt8",
        "per_channel": True,
        "op_types_to_quantize": ["MatMul"],
    }
    assert recipe["token_embedding"] == "float16 + Cast"
    assert recipe["stored_bytes"] == (out / "onnx" / "model.int8.onnx").stat().st_size
    stats = [json.loads(line) for line in (out / "activation-stats.jsonl").read_text().splitlines()]
    assert {"name", "act_max_mean", "act_max_max", "outlier_ratio"} <= set(stats[0])
    check = json.loads((out / "int8-check.json").read_text())
    assert check["n"] == 3 and 0 <= check["top1_agree"] <= 1 and "rmse_median" in check
