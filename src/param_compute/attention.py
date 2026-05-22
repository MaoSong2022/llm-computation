from loguru import logger
from .shared import linear_params, layernorm_params


def _gqa(cfg: dict, n_kv_heads: int | None = None, qk_norm=None) -> int:
    d_model = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = (
        cfg.get("num_key_value_heads", n_heads) if n_kv_heads is None else n_kv_heads
    )
    bias = cfg.get("bias", False)
    head_dim = d_model // n_heads

    if qk_norm:
        q_norm = layernorm_params(head_dim, bias)
        k_norm = layernorm_params(head_dim, bias)
    else:
        q_norm = 0
        k_norm = 0

    logger.info(
        "GQA config: {}".format(
            {
                "d_model": d_model,
                "n_heads": n_heads,
                "n_kv_heads": n_kv_heads,
                "bias": bias,
                "q_norm": q_norm,
                "k_norm": k_norm,
            }
        )
    )
    q = linear_params(d_model, n_heads * head_dim, bias)
    k = linear_params(d_model, n_kv_heads * head_dim, bias)
    v = linear_params(d_model, n_kv_heads * head_dim, bias)
    o = linear_params(n_heads * head_dim, d_model, bias)
    logger.debug(
        {
            "q": q,
            "k": k,
            "v": v,
            "o": o,
            "q_norm": q_norm,
            "k_norm": k_norm,
        }
    )
    return q + k + v + o + q_norm + k_norm


def _mha(cfg: dict, qk_norm=None) -> int:
    n_kv_heads = cfg["num_attention_heads"]
    return _gqa(cfg, n_kv_heads=n_kv_heads, qk_norm=qk_norm)


def _mqa(cfg: dict, qk_norm=None) -> int:
    return _gqa(cfg, n_kv_heads=1, qk_norm=qk_norm)


def _mla(cfg: dict, qk_norm=None) -> int:
    d_model = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    kv_lora_rank = cfg["kv_lora_rank"]
    q_lora_rank = cfg["q_lora_rank"]
    qk_rope_head_dim = cfg.get("qk_rope_head_dim", 0)
    v_head_dim = cfg.get("v_head_dim", d_model // n_heads)
    qk_nope_head_dim = cfg.get(
        "qk_nope_head_dim", (d_model // n_heads) - qk_rope_head_dim
    )
    bias = cfg.get("bias", False)

    logger.info(
        "MLA config: {}".format(
            {
                "d_model": d_model,
                "n_heads": n_heads,
                "q_lora_rank": q_lora_rank,
                "kv_lora_rank": kv_lora_rank,
                "qk_rope_head_dim": qk_rope_head_dim,
                "v_head_dim": v_head_dim,
                "qk_nope_head_dim": qk_nope_head_dim,
                "bias": bias,
            }
        )
    )

    if not q_lora_rank:
        q_up_proj = linear_params(
            d_model, (qk_nope_head_dim + qk_rope_head_dim) * n_heads, bias
        )
        q_layer_norm = 0
        q_down_proj = 0
        q_rope_proj = 0
    else:
        q_down_proj = linear_params(d_model, q_lora_rank, bias)
        q_layer_norm = layernorm_params(q_lora_rank, bias)
        q_up_proj = linear_params(q_lora_rank, qk_nope_head_dim * n_heads, bias)
        q_rope_proj = linear_params(q_lora_rank, qk_rope_head_dim * n_heads, bias)

    kv_down_proj = linear_params(d_model, kv_lora_rank, bias)
    k_layer_norm = layernorm_params(kv_lora_rank, bias)
    k_up_proj = linear_params(kv_lora_rank, v_head_dim * n_heads, bias)
    v_up_proj = linear_params(kv_lora_rank, v_head_dim * n_heads, bias)

    k_rope_proj = linear_params(d_model, qk_rope_head_dim, bias)
    # q_rope_proj = linear_params(d_model, qk_rope_head_dim * n_heads, bias)

    o = linear_params(v_head_dim * n_heads, d_model, bias)

    logger.debug(
        {
            "kv_down_proj": kv_down_proj,
            "k_layer_norm": k_layer_norm,
            "k_up_proj": k_up_proj,
            "v_up_proj": v_up_proj,
            "q_down_proj": q_down_proj,
            "q_layer_norm": q_layer_norm,
            "q_up_proj": q_up_proj,
            "k_rope_proj": k_rope_proj,
            "q_rope_proj": q_rope_proj,
            "o": o,
        }
    )

    return (
        kv_down_proj
        + k_layer_norm
        + k_up_proj
        + v_up_proj
        + q_down_proj
        + q_layer_norm
        + q_up_proj
        + k_rope_proj
        + q_rope_proj
        + o
    )


ATTENTION_VARIANTS = {
    "mha": _mha,
    "gqa": _gqa,
    "mqa": _mqa,
    "mla": _mla,
}
