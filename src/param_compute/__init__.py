import json
from loguru import logger

from .shared import embedding_params, layernorm_params, linear_params
from .attention import ATTENTION_VARIANTS
from .ffn import DENSE_FFN_VARIANTS
from .moe import MOE_VARIANTS

FFN_VARIANTS = {**DENSE_FFN_VARIANTS, **MOE_VARIANTS}

DENSE_COUNTERPARTS = {
    "moe": "gelu",
    "moe_gated": "silu",
    "deepseek_moe": "silu",
}


def _parse_ffn_spec(ffn: str) -> tuple[str, str]:
    if "+" in ffn:
        parts = ffn.split("+", 1)
        return parts[0], parts[1]
    return DENSE_COUNTERPARTS.get(ffn, ffn), ffn


def compute_params(
    config, attention="gqa", ffn="silu", tie_embeddings=None, num_dense_layers=None
):
    if isinstance(config, dict):
        cfg = config
    else:
        with open(config) as f:
            cfg = json.load(f)

    d_model = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    logger.info(f"number of layers: {n_layers}")
    vocab_size = cfg["vocab_size"]
    if tie_embeddings is None:
        tie_embeddings = cfg.get("tie_word_embeddings", False)

    emb = embedding_params(vocab_size, d_model)

    attn_fn = ATTENTION_VARIANTS.get(attention)
    if attn_fn is None:
        raise ValueError(
            f"Unknown attention variant: {attention}, available: {list(ATTENTION_VARIANTS)}"
        )

    dense_ffn, moe_ffn = _parse_ffn_spec(ffn)

    dense_ffn_fn = FFN_VARIANTS.get(dense_ffn)
    if dense_ffn_fn is None:
        raise ValueError(
            f"Unknown dense FFN variant: {dense_ffn}, available: {list(FFN_VARIANTS)}"
        )
    moe_ffn_fn = FFN_VARIANTS.get(moe_ffn)
    if moe_ffn_fn is None:
        raise ValueError(
            f"Unknown MoE FFN variant: {moe_ffn}, available: {list(FFN_VARIANTS)}"
        )

    num_dense = (
        cfg.get("num_dense_layers", 0) if num_dense_layers is None else num_dense_layers
    )
    num_moe = n_layers - num_dense

    n_attn_params = attn_fn(cfg)
    dense_total, dense_activated = dense_ffn_fn(cfg) if num_dense > 0 else (0, 0)
    moe_total, moe_activated = moe_ffn_fn(cfg) if num_moe > 0 else (0, 0)

    logger.debug(
        {
            "dense_total": dense_total,
            "dense_activated": dense_activated,
            "moe_total": moe_total,
            "moe_activated": moe_activated,
        }
    )

    norm_type = cfg.get("normalization", "rmsnorm")
    bias = cfg.get("bias", False)

    ln_per_layer = 2 * layernorm_params(d_model, bias, norm_type)
    final_ln = layernorm_params(d_model, bias, norm_type)
    lm_head = 0 if tie_embeddings else embedding_params(vocab_size, d_model)

    n_ffn_total = num_dense * dense_total + num_moe * moe_total
    n_ffn_activated = num_dense * dense_activated + num_moe * moe_activated

    totals = {
        "embedding": emb + lm_head,
        "attention": n_layers * n_attn_params,
        "layernorm": n_layers * ln_per_layer + final_ln,
        "ffn": n_ffn_total,
    }
    totals["total_params"] = sum(totals.values())
    totals["ffn_activated"] = n_ffn_activated
    totals["activated_total"] = totals["total_params"] - n_ffn_total + n_ffn_activated
    return totals
