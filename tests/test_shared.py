import pytest
from src.param_compute.shared import linear_params, embedding_params, layernorm_params


class TestLinearParams:
    def test_without_bias(self):
        assert linear_params(10, 20) == 200
        assert linear_params(4096, 4096) == 16_777_216

    def test_with_bias(self):
        assert linear_params(10, 20, bias=True) == 200 + 20
        assert linear_params(4096, 4096, bias=True) == 16_777_216 + 4096

    def test_zero_input(self):
        assert linear_params(0, 100) == 0
        assert linear_params(0, 100, bias=True) == 100


class TestEmbeddingParams:
    def test_basic(self):
        assert embedding_params(100, 512) == 51_200
        assert embedding_params(151936, 4096) == 622_329_856


class TestLayerNormParams:
    def test_rmsnorm(self):
        assert layernorm_params(4096, norm_type="rmsnorm") == 4096
        assert layernorm_params(4096, bias=False, norm_type="rmsnorm") == 4096
        assert layernorm_params(4096, bias=True, norm_type="rmsnorm") == 4096

    def test_layernorm_without_bias(self):
        assert layernorm_params(4096, bias=False, norm_type="layernorm") == 4096

    def test_layernorm_with_bias(self):
        assert layernorm_params(4096, bias=True, norm_type="layernorm") == 8192

    def test_layer_norm_alias(self):
        assert layernorm_params(4096, bias=True, norm_type="layer_norm") == 8192

    def test_unknown_norm_type(self):
        with pytest.raises(ValueError, match="Unsupported norm"):
            layernorm_params(4096, norm_type="unknown")

    def test_default_norm_type(self):
        assert layernorm_params(512) == 512
