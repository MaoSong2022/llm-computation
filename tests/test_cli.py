import argparse

import pytest
from compute_model_params import (
    detect_attention,
    detect_ffn,
    detect_num_dense_layers,
    detect_qk_norm,
    _validate_ffn,
)
from src.param_compute import _parse_ffn_spec


class TestDetectAttention:
    def test_mla(self):
        cfg = {"kv_lora_rank": 512}
        assert detect_attention(cfg) == "mla"

    def test_mqa(self):
        cfg = {"num_key_value_heads": 1}
        assert detect_attention(cfg) == "mqa"

    def test_gqa(self):
        cfg = {"num_key_value_heads": 8}
        assert detect_attention(cfg) == "gqa"

    def test_mha(self):
        cfg = {}
        assert detect_attention(cfg) == "mha"

    def test_mha_with_heads(self):
        cfg = {"num_attention_heads": 32}
        assert detect_attention(cfg) == "mha"


class TestDetectFFN:
    def test_deepseek_moe(self):
        cfg = {"n_routed_experts": 64}
        assert detect_ffn(cfg) == "deepseek_moe"

    def test_deepseek_moe_alt(self):
        cfg = {"num_routed_experts": 64}
        assert detect_ffn(cfg) == "deepseek_moe"

    def test_moe_gated(self):
        cfg = {"num_local_experts": 8}
        assert detect_ffn(cfg) == "moe_gated"

    def test_moe_gated_alt(self):
        cfg = {"num_experts": 8}
        assert detect_ffn(cfg) == "moe_gated"

    def test_silu_default(self):
        cfg = {}
        assert detect_ffn(cfg) == "silu"


class TestDetectQKNorm:
    def test_qk_norm(self):
        assert detect_qk_norm({"qk_norm": True}) is True
        assert detect_qk_norm({"qk_norm": False}) is False

    def test_use_qk_norm(self):
        assert detect_qk_norm({"use_qk_norm": True}) is True
        assert detect_qk_norm({"use_qk_norm": False}) is False

    def test_no_qk_norm(self):
        assert detect_qk_norm({}) is False


class TestDetectNumDenseLayers:
    def test_present(self):
        assert detect_num_dense_layers({"num_dense_layers": 4}) == 4

    def test_absent(self):
        assert detect_num_dense_layers({}) == 0


class TestValidateFFN:
    def test_valid_variants(self):
        for v in ["silu", "swiglu", "gelu", "relu", "moe", "moe_gated", "deepseek_moe"]:
            assert _validate_ffn(v) == v

    def test_hybrid(self):
        assert _validate_ffn("silu+deepseek_moe") == "silu+deepseek_moe"

    def test_invalid(self):
        with pytest.raises((argparse.ArgumentTypeError, ValueError)):
            _validate_ffn("invalid")


class TestParseFFNSpec:
    def test_dense_only(self):
        assert _parse_ffn_spec("silu") == ("silu", "silu")
        assert _parse_ffn_spec("gelu") == ("gelu", "gelu")

    def test_moe_only(self):
        assert _parse_ffn_spec("moe") == ("gelu", "moe")
        assert _parse_ffn_spec("moe_gated") == ("silu", "moe_gated")
        assert _parse_ffn_spec("deepseek_moe") == ("silu", "deepseek_moe")

    def test_hybrid(self):
        assert _parse_ffn_spec("silu+deepseek_moe") == ("silu", "deepseek_moe")
        assert _parse_ffn_spec("gelu+moe") == ("gelu", "moe")
