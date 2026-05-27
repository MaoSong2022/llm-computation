import pytest
from src.param_compute.moe import _moe, _moe_gated, _deepseek_moe, MOE_VARIANTS


class TestMoE:
    def test_basic(self):
        cfg = {
            "hidden_size": 4096,
            "moe_intermediate_size": 1536,
            "num_experts": 128,
            "num_experts_per_tok": 8,
        }
        total, activated = _moe(cfg)
        d_ff = 1536
        expert = 3 * 4096 * d_ff
        router = 4096 * 128
        expected_total = 128 * expert + router
        expected_activated = 8 * expert + router
        assert total == expected_total
        assert activated == expected_activated

    def test_with_alias_fields(self):
        cfg = {
            "hidden_size": 2048,
            "moe_intermediate_size": 768,
            "num_local_experts": 64,
            "top_k": 4,
        }
        total, activated = _moe(cfg)
        d_ff = 768
        expert = 3 * 2048 * d_ff
        router = 2048 * 64
        assert total == 64 * expert + router
        assert activated == 4 * expert + router

    def test_default_intermediate_size(self):
        cfg = {
            "hidden_size": 1024,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        }
        total, _ = _moe(cfg)
        d_ff = 4 * 1024
        expert = 3 * 1024 * d_ff
        router = 1024 * 8
        assert total == 8 * expert + router


class TestMoEGated:
    def test_basic(self):
        cfg = {
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        }
        total, activated = _moe_gated(cfg)
        d_ff = 12288
        expert = 3 * 4096 * d_ff
        router = 4096 * 8
        assert total == 8 * expert + router
        assert activated == 2 * expert + router


class TestDeepSeekMoE:
    def test_basic(self):
        cfg = {
            "hidden_size": 2048,
            "moe_intermediate_size": 768,
            "num_routed_experts": 128,
            "num_experts_per_tok": 8,
        }
        total, activated = _deepseek_moe(cfg)
        d_ff = 768
        expert = 3 * 2048 * d_ff
        shared = 1 * expert
        routed = 128 * expert
        router = 2048 * 128
        assert total == shared + routed + router
        assert activated == shared + 8 * expert + router

    def test_with_shared_experts(self):
        cfg = {
            "hidden_size": 2048,
            "moe_intermediate_size": 768,
            "num_routed_experts": 128,
            "num_experts_per_tok": 8,
            "num_shared_experts": 2,
        }
        total, activated = _deepseek_moe(cfg)
        d_ff = 768
        expert = 3 * 2048 * d_ff
        shared = 2 * expert
        routed = 128 * expert
        router = 2048 * 128
        assert total == shared + routed + router
        assert activated == shared + 8 * expert + router

    def test_with_n_routed_experts(self):
        cfg = {
            "hidden_size": 2048,
            "moe_intermediate_size": 768,
            "n_routed_experts": 64,
            "num_shared_experts": 1,
            "num_experts_per_tok": 6,
        }
        total, activated = _deepseek_moe(cfg)
        d_ff = 768
        expert = 3 * 2048 * d_ff
        assert total == 1 * expert + 64 * expert + 2048 * 64
        assert activated == 1 * expert + 6 * expert + 2048 * 64

    def test_uses_intermediate_size_fallback(self):
        cfg = {
            "hidden_size": 2048,
            "intermediate_size": 4096,
            "num_routed_experts": 64,
            "num_experts_per_tok": 8,
        }
        total, _ = _deepseek_moe(cfg)
        d_ff = 4096
        expert = 3 * 2048 * d_ff
        assert total > 0


class TestMOEVariants:
    def test_mapping(self):
        assert MOE_VARIANTS["moe"] is _moe
        assert MOE_VARIANTS["moe_gated"] is _moe_gated
        assert MOE_VARIANTS["deepseek_moe"] is _deepseek_moe
