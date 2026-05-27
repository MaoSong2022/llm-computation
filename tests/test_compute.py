import json
from pathlib import Path

import pytest

from src.param_compute import compute_params


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def qwen_config(model_name):
    with open(DATA_DIR / "qwen.json") as f:
        data = json.load(f)
    for entry in data:
        if entry["model"] == model_name:
            return entry
    raise ValueError(f"Model {model_name} not found in qwen.json")


class TestDenseModels:
    def test_qwen3_0_6B(self):
        entry = qwen_config("Qwen/Qwen3-0.6B")
        result = compute_params(entry["config"])
        self._assert_matches(entry, result)

    def test_qwen3_1_7B(self):
        entry = qwen_config("Qwen/Qwen3-1.7B")
        result = compute_params(entry["config"])
        self._assert_matches(entry, result)

    def test_qwen3_4B(self):
        entry = qwen_config("Qwen/Qwen3-4B")
        result = compute_params(entry["config"])
        self._assert_matches(entry, result)

    def test_qwen3_8B(self):
        entry = qwen_config("Qwen/Qwen3-8B")
        result = compute_params(entry["config"])
        self._assert_matches(entry, result)

    def test_qwen3_14B(self):
        entry = qwen_config("Qwen/Qwen3-14B")
        result = compute_params(entry["config"])
        self._assert_matches(entry, result)

    def test_qwen3_32B(self):
        entry = qwen_config("Qwen/Qwen3-32B")
        result = compute_params(entry["config"])
        self._assert_matches(entry, result)

    def _assert_matches(self, entry, result):
        tol = 0.02
        assert abs(result["total_params"] / 1e9 - entry["total_params_billion"]) < tol
        assert (
            abs(result["activated_total"] / 1e9 - entry["active_params_billion"]) < tol
        )
        key_map = {"embedding": "Embeddings", "attention": "Attention", "ffn": "FFN", "layernorm": "Norms"}
        for rkey, pname in key_map.items():
            p = next(x for x in entry["parts"] if x["name"] == pname)
            assert abs(result[rkey] / 1e9 - p["params_billion"]) < tol


class TestMoEModels:
    def test_qwen3_30B_A3B(self):
        entry = qwen_config("Qwen/Qwen3-30B-A3B")
        cfg = dict(entry["config"])
        cfg["num_routed_experts"] = cfg.get("num_experts", 128)
        cfg["num_shared_experts"] = 0
        result = compute_params(cfg, ffn="silu+deepseek_moe")
        self._assert_matches(entry, result)

    def test_qwen3_235B_A22B(self):
        entry = qwen_config("Qwen/Qwen3-235B-A22B")
        cfg = dict(entry["config"])
        cfg["num_routed_experts"] = cfg.get("num_experts", 128)
        cfg["num_shared_experts"] = 0
        result = compute_params(cfg, ffn="silu+deepseek_moe")
        self._assert_matches(entry, result)

    def test_qwen3_30B_as_moe_gated(self):
        entry = qwen_config("Qwen/Qwen3-30B-A3B")
        result = compute_params(entry["config"], ffn="moe_gated")
        assert result["total_params"] > 0
        assert result["activated_total"] < result["total_params"]

    def test_qwen3_235B_as_moe_gated(self):
        entry = qwen_config("Qwen/Qwen3-235B-A22B")
        result = compute_params(entry["config"], ffn="moe_gated")
        assert result["total_params"] > 0
        assert result["activated_total"] < result["total_params"]

    def _assert_matches(self, entry, result):
        tol = 0.02
        assert abs(result["total_params"] / 1e9 - entry["total_params_billion"]) < tol
        assert (
            abs(result["activated_total"] / 1e9 - entry["active_params_billion"]) < tol
        )
        key_map = {"embedding": "Embeddings", "attention": "Attention", "ffn": "FFN", "layernorm": "Norms"}
        for rkey, pname in key_map.items():
            p = next(x for x in entry["parts"] if x["name"] == pname)
            assert abs(result[rkey] / 1e9 - p["params_billion"]) < tol


class TestEdgeCases:
    def test_tied_embeddings(self):
        cfg = {
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "vocab_size": 32000,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "intermediate_size": 2048,
            "tie_word_embeddings": True,
        }
        result = compute_params(cfg)
        tied = compute_params(cfg, tie_embeddings=True)
        untied = compute_params(cfg, tie_embeddings=False)
        assert tied["embedding"] < untied["embedding"]
        assert tied["total_params"] < untied["total_params"]
        assert result == tied

    def test_untied_embeddings(self):
        cfg = {
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "vocab_size": 32000,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "intermediate_size": 2048,
        }
        untied = compute_params(cfg, tie_embeddings=False)
        d_model = 512
        expected_emb = 32000 * d_model * 2
        assert untied["embedding"] == expected_emb

    def test_with_qk_norm(self):
        cfg = {
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "vocab_size": 32000,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "intermediate_size": 2048,
        }
        with_norm = compute_params(cfg, qk_norm=True)
        without_norm = compute_params(cfg, qk_norm=False)
        assert with_norm["attention"] > without_norm["attention"]
        diff = with_norm["attention"] - without_norm["attention"]
        assert diff == 6 * 2 * 64

    def test_layernorm_type(self):
        cfg = {
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "vocab_size": 32000,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "intermediate_size": 2048,
            "normalization": "layernorm",
            "bias": True,
        }
        result = compute_params(cfg)
        d_model = 512
        expected_ln = 6 * 2 * (d_model * 2) + (d_model * 2)
        assert result["layernorm"] == expected_ln

    def test_hybrid_ffn(self):
        cfg = {
            "hidden_size": 1024,
            "num_hidden_layers": 8,
            "vocab_size": 32000,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 4096,
            "num_dense_layers": 2,
            "num_routed_experts": 4,
            "num_shared_experts": 1,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 2048,
        }
        result = compute_params(cfg, ffn="silu+deepseek_moe")
        dense_ffn = 3 * 1024 * 4096
        d_ff_moe = 2048
        expert_moe = 3 * 1024 * d_ff_moe
        moe_total = (1 * expert_moe) + (4 * expert_moe) + (1024 * 4)
        assert result["ffn"] == 2 * dense_ffn + 6 * moe_total

    def test_moe_activated_less_than_total(self):
        cfg = {
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_experts": 16,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 512,
        }
        result = compute_params(cfg, ffn="moe")
        assert result["ffn_activated"] < result["ffn"]
        assert result["activated_total"] < result["total_params"]

    def test_missing_num_dense_layers_defaults_to_zero(self):
        cfg = {
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "vocab_size": 32000,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "intermediate_size": 2048,
        }
        result = compute_params(cfg)
        d_ff = 2048
        expected_ffn = 6 * (3 * 512 * d_ff)
        assert result["ffn"] == expected_ffn


class TestConfigLoading:
    def test_accepts_dict(self):
        cfg = {
            "hidden_size": 512,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "intermediate_size": 2048,
        }
        result = compute_params(cfg)
        assert result["total_params"] > 0

    def test_accepts_json_file_path(self, tmp_path):
        cfg = {
            "hidden_size": 512,
            "num_hidden_layers": 4,
            "vocab_size": 32000,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "intermediate_size": 2048,
        }
        p = tmp_path / "config.json"
        with open(p, "w") as f:
            json.dump(cfg, f)
        result = compute_params(str(p))
        assert result["total_params"] > 0

    def test_invalid_attention(self):
        cfg = {"hidden_size": 512, "num_hidden_layers": 2, "vocab_size": 1000}
        with pytest.raises(ValueError, match="Unknown attention variant"):
            compute_params(cfg, attention="invalid")

    def test_invalid_ffn(self):
        cfg = {"hidden_size": 512, "num_hidden_layers": 2, "vocab_size": 1000}
        with pytest.raises(ValueError, match="Unknown dense FFN variant"):
            compute_params(cfg, ffn="invalid")
