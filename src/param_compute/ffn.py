from loguru import logger
from .shared import linear_params


def _dense_gated(cfg: dict) -> tuple[int, int]:
    d_model = cfg["hidden_size"]
    d_ff = cfg.get("intermediate_size", 4 * d_model)
    bias = cfg.get("bias", False)

    logger.info(
        "Dense gated FFN config: {}".format(
            {"d_model": d_model, "d_ff": d_ff, "bias": bias}
        )
    )
    gate = linear_params(d_model, d_ff, bias)
    up = linear_params(d_model, d_ff, bias)
    down = linear_params(d_ff, d_model, bias)
    total = gate + up + down
    return total, total


def _dense(cfg: dict) -> tuple[int, int]:
    d_model = cfg["hidden_size"]
    d_ff = cfg.get("intermediate_size", 4 * d_model)
    bias = cfg.get("bias", False)
    logger.info(
        "Dense FFN config: {}".format({"d_model": d_model, "d_ff": d_ff, "bias": bias})
    )
    total = linear_params(d_model, d_ff, bias) + linear_params(d_ff, d_model, bias)
    return total, total


DENSE_FFN_VARIANTS = {
    "silu": _dense_gated,
    "swiglu": _dense_gated,
    "gelu": _dense,
    "relu": _dense,
}
