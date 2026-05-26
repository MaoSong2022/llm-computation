from loguru import logger

from .shared import linear_params
from .ffn import DENSE_FFN_VARIANTS


def _get_n_experts(cfg: dict) -> int:
    v = cfg.get("num_local_experts")
    if v is None:
        v = cfg.get("num_experts", 0)
    return v


def _get_activated_experts(cfg: dict) -> int:
    v = cfg.get("num_experts_per_tok")
    if v is None:
        v = cfg.get("top_k", 1)
    return v


def _moe(cfg: dict) -> tuple[int, int]:
    d_model = cfg["hidden_size"]
    d_ff = cfg.get("moe_intermediate_size", 4 * d_model)
    n_experts = _get_n_experts(cfg)
    top_k = _get_activated_experts(cfg)
    bias = cfg.get("bias", False)

    logger.info(
        "MoE FFN config: {}".format(
            {
                "d_model": d_model,
                "d_ff": d_ff,
                "n_experts": n_experts,
                "top_k": top_k,
                "bias": bias,
            }
        )
    )
    ffn = DENSE_FFN_VARIANTS["silu"]
    expert, _ = ffn(
        {
            "hidden_size": d_model,
            "intermediate_size": d_ff,
            "bias": bias,
        }
    )
    router = linear_params(d_model, n_experts)
    total = n_experts * expert + router
    activated = top_k * expert + router
    return total, activated


def _moe_gated(cfg: dict) -> tuple[int, int]:
    d_model = cfg["hidden_size"]
    d_ff = cfg.get("intermediate_size", 4 * d_model)
    n_experts = _get_n_experts(cfg)
    top_k = _get_activated_experts(cfg)
    bias = cfg.get("bias", False)

    logger.info(
        "MoE gated FFN config: {}".format(
            {
                "d_model": d_model,
                "d_ff": d_ff,
                "n_experts": n_experts,
                "top_k": top_k,
                "bias": bias,
            }
        )
    )
    ffn = DENSE_FFN_VARIANTS["silu"]

    expert, _ = ffn(
        {
            "hidden_size": d_model,
            "intermediate_size": d_ff,
            "bias": bias,
        }
    )
    router = linear_params(d_model, n_experts)
    total = n_experts * expert + router
    activated = top_k * expert + router
    return total, activated


def _deepseek_moe(cfg: dict) -> tuple[int, int]:
    d_model = cfg["hidden_size"]
    d_ff = cfg.get("moe_intermediate_size")
    if d_ff is None:
        d_ff = cfg.get("intermediate_size", 4 * d_model)
    n_routed = cfg.get("n_routed_experts") or cfg.get("num_routed_experts")
    n_shared = cfg.get("n_shared_experts")
    if n_shared is None:
        n_shared = cfg.get("num_shared_experts", 1)
    top_k = _get_activated_experts(cfg)
    bias = cfg.get("bias", False)

    logger.info(
        "DeepSeek MoE FFN config: {}".format(
            {
                "d_model": d_model,
                "d_ff": d_ff,
                "n_routed": n_routed,
                "n_shared": n_shared,
                "top_k": top_k,
                "bias": bias,
            }
        )
    )

    shared_expert = 2 * linear_params(d_model, d_ff, bias) + linear_params(
        d_ff, d_model, bias
    )
    shared = n_shared * shared_expert
    routed = n_routed * shared_expert
    router = linear_params(d_model, n_routed)
    total = shared + routed + router
    activated = shared + top_k * shared_expert + router
    return total, activated


MOE_VARIANTS = {
    "moe": _moe,
    "moe_gated": _moe_gated,
    "deepseek_moe": _deepseek_moe,
}
