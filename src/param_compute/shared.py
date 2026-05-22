from loguru import logger


def linear_params(in_f: int, out_f: int, bias: bool = False) -> int:
    return in_f * out_f + (out_f if bias else 0)


def embedding_params(vocab_size: int, d_model: int) -> int:
    logger.info(
        f"Embedding config: {{'vocab_size': {vocab_size}, 'd_model': {d_model}}}"
    )
    return vocab_size * d_model


def layernorm_params(
    d_model: int, bias: bool = False, norm_type: str = "rmsnorm"
) -> int:
    logger.info(
        f"LayerNorm config: {{'d_model': {d_model}, 'bias': {bias}, 'norm_type': {norm_type}}}"
    )
    if norm_type == "rmsnorm":
        return d_model
    elif norm_type in ("layernorm", "layer_norm"):
        return d_model * (2 if bias else 1)
    raise ValueError(f"Unsupported norm: {norm_type}")
