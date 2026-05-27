import pytest
from src.param_compute.attention import (
    _mha,
    _gqa,
    _mqa,
    _mla,
    ATTENTION_VARIANTS,
)


class TestGQA:
    def test_basic(self):
        cfg = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
        n_params = _gqa(cfg)
        q = 4096 * (32 * 128)
        k = 4096 * (8 * 128)
        v = 4096 * (8 * 128)
        o = (32 * 128) * 4096
        assert n_params == q + k + v + o

    def test_with_bias(self):
        cfg = {
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "bias": True,
        }
        n_params = _gqa(cfg)
        q = 1024 * (16 * 64) + (16 * 64)
        k = 1024 * (4 * 64) + (4 * 64)
        v = 1024 * (4 * 64) + (4 * 64)
        o = (16 * 64) * 1024 + 1024
        assert n_params == q + k + v + o

    def test_with_qk_norm(self):
        cfg = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
        n_params = _gqa(cfg, qk_norm=True)
        q = 4096 * (32 * 128)
        k = 4096 * (8 * 128)
        v = 4096 * (8 * 128)
        o = (32 * 128) * 4096
        q_norm = 128
        k_norm = 128
        assert n_params == q + k + v + o + q_norm + k_norm

    def test_default_head_dim(self):
        cfg = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }
        n_params = _gqa(cfg)
        head_dim = 4096 // 32
        q = 4096 * (32 * head_dim)
        k = 4096 * (8 * head_dim)
        v = 4096 * (8 * head_dim)
        o = (32 * head_dim) * 4096
        assert n_params == q + k + v + o


class TestMHA:
    def test_delegates_to_gqa(self):
        cfg = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "head_dim": 128,
        }
        assert _mha(cfg) == _gqa(cfg, n_kv_heads=32)


class TestMQA:
    def test_single_kv_head(self):
        cfg = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "head_dim": 128,
        }
        n_params = _mqa(cfg)
        q = 4096 * (32 * 128)
        k = 4096 * (1 * 128)
        v = 4096 * (1 * 128)
        o = (32 * 128) * 4096
        assert n_params == q + k + v + o


class TestMLA:
    def test_with_q_lora_rank(self):
        cfg = {
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "qk_nope_head_dim": 128,
        }
        n_params = _mla(cfg)
        kv_down = 4096 * 512
        k_ln = 512
        k_up = 512 * (128 * 64)
        v_up = 512 * (128 * 64)
        q_down = 4096 * 1536
        q_ln = 1536
        q_up = 1536 * (128 * 64)
        q_rope = 1536 * (64 * 64)
        k_rope = 4096 * 64
        o = (128 * 64) * 4096
        expected = (
            kv_down + k_ln + k_up + v_up + q_down + q_ln + q_up + q_rope + k_rope + o
        )
        assert n_params == expected

    def test_without_q_lora_rank(self):
        cfg = {
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "kv_lora_rank": 512,
            "q_lora_rank": 0,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "qk_nope_head_dim": 128,
        }
        n_params = _mla(cfg)
        kv_down = 4096 * 512
        k_ln = 512
        k_up = 512 * (128 * 64)
        v_up = 512 * (128 * 64)
        q_up = 4096 * (128 + 64) * 64
        k_rope = 4096 * 64
        o = (128 * 64) * 4096
        expected = kv_down + k_ln + k_up + v_up + q_up + k_rope + o
        assert n_params == expected


class TestAttentionVariants:
    def test_all_variants_present(self):
        assert set(ATTENTION_VARIANTS) == {"mha", "gqa", "mqa", "mla"}
