import json
from pathlib import Path

import pytest

from src.param_compute import compute_params

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_series(name: str) -> list[dict]:
    with open(DATA_DIR / f"{name}.json") as f:
        return json.load(f)


def entry_for(series: str, model: str) -> dict:
    for e in load_series(series):
        if e["model"] == model:
            return e
    raise ValueError(f"{model} not found in {series}.json")


def _validate_or_skip(entry, result):
    """Assert computed result matches reference; skip if ref values unset."""
    if entry["total_params_billion"] is None:
        pytest.skip("Reference value total_params_billion not yet set in data file")

    assert abs(result["total_params"] == entry["total_params"])
    assert abs(result["activated_total"] == entry["activated_total"])

    key_map = {
        "embedding": "Embeddings",
        "attention": "Attention",
        "ffn": "FFN",
        "layernorm": "Norms",
    }
    for rkey, pname in key_map.items():
        part = next(p for p in entry["parts"] if p["name"] == pname)
        if part["params_billion"] is None:
            pytest.skip(
                f"Reference value {pname}.params_billion not yet set in data file"
            )
        assert abs(result[rkey] / 1e9 - part["params_billion"]) < 0.02


class TestDeepSeekSeries:
    """DeepSeek V2/V3/R1 series — MLA + DeepSeek MoE.

    Steps to activate:
      1. Fill in the expected parameter values in data/deepseek.json
         (total_params_billion, active_params_billion, parts[].params_billion)
      2. Re-run — the skip will turn into a real assertion.
    """

    def test_deepseek_v3(self):
        entry = entry_for("deepseek", "deepseek-ai/DeepSeek-V3")
        result = compute_params(
            entry["config"],
            attention="mla",
            ffn="silu+deepseek_moe",
            num_dense_layers=entry["config"]["num_dense_layers"],
        )
        _validate_or_skip(entry, result)

    def test_deepseek_v2(self):
        entry = entry_for("deepseek", "deepseek-ai/DeepSeek-V2")
        result = compute_params(
            entry["config"],
            attention="mla",
            ffn="silu+deepseek_moe",
            num_dense_layers=entry["config"]["num_dense_layers"],
        )
        _validate_or_skip(entry, result)

    def test_deepseek_r1(self):
        entry = entry_for("deepseek", "deepseek-ai/DeepSeek-R1")
        result = compute_params(
            entry["config"],
            attention="mla",
            ffn="silu+deepseek_moe",
            num_dense_layers=entry["config"]["num_dense_layers"],
        )
        _validate_or_skip(entry, result)

    def test_deepseek_v3_runs(self):
        entry = entry_for("deepseek", "deepseek-ai/DeepSeek-V3")
        result = compute_params(
            entry["config"],
            attention="mla",
            ffn="silu+deepseek_moe",
            num_dense_layers=3,
        )
        assert result["total_params"] > 0
        assert result["activated_total"] < result["total_params"]

    def test_deepseek_v2_runs(self):
        entry = entry_for("deepseek", "deepseek-ai/DeepSeek-V2")
        result = compute_params(
            entry["config"],
            attention="mla",
            ffn="silu+deepseek_moe",
            num_dense_layers=2,
        )
        assert result["total_params"] > 0
        assert result["activated_total"] < result["total_params"]


class TestModelSeriesSmoke:
    """Smoke tests for ALL model series in data/*.json."""

    @pytest.mark.parametrize(
        "series",
        [
            "qwen",
            "deepseek",
        ],
    )
    def test_series_runs(self, series):
        entries = load_series(series)
        for entry in entries:
            cfg = dict(entry["config"])
            ffn = entry.get("ffn", "silu")
            attention = entry.get("attention", "gqa").lower()
            num_dense = cfg.get("num_dense_layers")
            num_dense_kw = (
                {"num_dense_layers": num_dense} if num_dense is not None else {}
            )

            if "kv_lora_rank" in cfg:
                attention = "mla"

            result = compute_params(cfg, attention=attention, ffn=ffn, **num_dense_kw)
            assert result["total_params"] > 0
            assert result["activated_total"] > 0
