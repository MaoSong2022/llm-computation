import pytest
from src.param_compute.ffn import _dense_gated, _dense, DENSE_FFN_VARIANTS


class TestDenseGated:
    def test_basic(self):
        cfg = {"hidden_size": 4096, "intermediate_size": 12288}
        total, activated = _dense_gated(cfg)
        gate = 4096 * 12288
        up = 4096 * 12288
        down = 12288 * 4096
        expected = gate + up + down
        assert total == expected
        assert activated == expected

    def test_with_bias(self):
        cfg = {"hidden_size": 1024, "intermediate_size": 4096, "bias": True}
        total, activated = _dense_gated(cfg)
        gate = 1024 * 4096 + 4096
        up = 1024 * 4096 + 4096
        down = 4096 * 1024 + 1024
        expected = gate + up + down
        assert total == expected

    def test_default_intermediate_size(self):
        cfg = {"hidden_size": 512}
        total, _ = _dense_gated(cfg)
        d_ff = 4 * 512
        expected = 3 * 512 * d_ff
        assert total == expected


class TestDense:
    def test_basic(self):
        cfg = {"hidden_size": 4096, "intermediate_size": 12288}
        total, activated = _dense(cfg)
        expected = 4096 * 12288 + 12288 * 4096
        assert total == expected
        assert activated == expected

    def test_with_bias(self):
        cfg = {"hidden_size": 1024, "intermediate_size": 4096, "bias": True}
        total, _ = _dense(cfg)
        expected = (1024 * 4096 + 4096) + (4096 * 1024 + 1024)
        assert total == expected

    def test_default_intermediate_size(self):
        cfg = {"hidden_size": 512}
        total, _ = _dense(cfg)
        d_ff = 4 * 512
        expected = 2 * 512 * d_ff
        assert total == expected


class TestDenseFFNVariants:
    def test_mapping(self):
        assert DENSE_FFN_VARIANTS["silu"] is _dense_gated
        assert DENSE_FFN_VARIANTS["swiglu"] is _dense_gated
        assert DENSE_FFN_VARIANTS["gelu"] is _dense
        assert DENSE_FFN_VARIANTS["relu"] is _dense
