#!/usr/bin/env python3
import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from loguru import logger
from src.param_compute import compute_params


HF_CONFIG_URL = "https://huggingface.co/{model}/raw/main/config.json"


def load_config(source: str) -> dict:
    if Path(source).exists():
        with open(source) as f:
            return json.load(f)
    url = HF_CONFIG_URL.format(model=source)

    with urllib.request.urlopen(url) as resp:
        body = resp.read().decode()
        return json.loads(body)


def detect_attention(cfg: dict) -> str:
    if "kv_lora_rank" in cfg:
        return "mla"
    n_kv = cfg.get("num_key_value_heads")
    if n_kv is not None:
        return "mqa" if n_kv == 1 else "gqa"
    return "mha"


def detect_ffn(cfg: dict) -> str:
    if "n_routed_experts" in cfg or "num_routed_experts" in cfg:
        return "deepseek_moe"
    if "num_local_experts" in cfg or "num_experts" in cfg:
        return "moe_gated"
    return "silu"


def detect_num_dense_layers(cfg: dict) -> int:
    return cfg.get("num_dense_layers", 0)


def detect_qk_norm(cfg: dict) -> bool:
    return bool(cfg.get("qk_norm", False) or cfg.get("use_qk_norm", False))


FFN_CHOICES = ["silu", "swiglu", "gelu", "relu", "moe", "moe_gated", "deepseek_moe"]


def _validate_ffn(val: str) -> str:
    parts = val.split("+")
    for p in parts:
        if p not in FFN_CHOICES:
            raise argparse.ArgumentTypeError(
                f"Unknown FFN variant '{p}' in '{val}'. "
                f"Available variants: {FFN_CHOICES}. "
                f"Use 'dense+moe' format for hybrid (e.g. silu+moe_gated)."
            )
    return val


def export_result(result: dict, cfg: dict, args: argparse.Namespace, attention: str, ffn: str) -> None:
    model_name = cfg.get("_name_or_path", args.model)
    family = "MoE" if any(k in ffn for k in ["moe", "moe_gated", "deepseek_moe"]) else "Dense"

    parts = [
        {"name": "Embeddings", "params_billion": round(result["embedding"] / 1e9, 2),
         "description": "Token embedding and output projection"},
        {"name": "Attention", "params_billion": round(result["attention"] / 1e9, 2),
         "description": "Q/K/V/O projections across all layers"},
        {"name": "FFN", "params_billion": round(result["ffn"] / 1e9, 2),
         "description": "Feed-forward blocks"},
        {"name": "Norms", "params_billion": round(result["layernorm"] / 1e9, 2),
         "description": "Layer norms and auxiliary parameters"},
    ]

    record = {
        "model": model_name,
        "year": args.year,
        "family": family,
        "attention": attention.upper(),
        "total_params_billion": round(result["total_params"] / 1e9, 2),
        "active_params_billion": round(result["activated_total"] / 1e9, 2),
        "config": cfg,
        "parts": parts,
    }

    out = Path(args.export)
    if out.exists():
        with open(out) as f:
            data = json.load(f)
    else:
        data = []
    data.append(record)

    with open(out, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    logger.info(f"Exported result to {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute parameter count of a HuggingFace model from its config."
    )
    parser.add_argument(
        "--model",
        "-m",
        help="HuggingFace model name (e.g. deepseek-ai/DeepSeek-67B) or local path to config.json",
    )
    parser.add_argument(
        "--attention",
        "-a",
        choices=["mha", "gqa", "mqa", "mla"],
        help="Attention variant (auto-detected if omitted)",
    )
    parser.add_argument(
        "--ffn",
        "-f",
        type=_validate_ffn,
        default="silu",
        help="FFN variant (auto-detected if omitted). Use 'dense+moe' format for hybrid (e.g. silu+moe_gated).",
    )
    parser.add_argument(
        "--num-dense-layers",
        type=int,
        default=None,
        help="Number of initial dense layers (overrides config's num_dense_layers).",
    )
    parser.add_argument(
        "--qk-norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalization for query and key (auto-detected if omitted)",
    )
    parser.add_argument(
        "--tie-embeddings",
        "-t",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override tie_word_embeddings, default to False",
    )
    parser.add_argument(
        "--export",
        "-e",
        type=str,
        help="Export results to a JSON file (appends to existing file if it exists)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year of the model (default: 2025)",
    )

    args = parser.parse_args()

    cfg = load_config(args.model)
    logger.info(f"Model: {cfg.get('_name_or_path', args.model)}")
    if "architectures" in cfg:
        logger.info(f"Architecture: {cfg['architectures']}")

    attention = args.attention or detect_attention(cfg)
    ffn = args.ffn or detect_ffn(cfg)
    num_dense = (
        detect_num_dense_layers(cfg)
        if args.num_dense_layers is None
        else args.num_dense_layers
    )
    qk_norm = detect_qk_norm(cfg) if args.qk_norm is None else args.qk_norm
    logger.info(
        f"Detected / using -> attention: {attention}, ffn: {ffn}, num_dense_layers: {num_dense}, qk_norm: {qk_norm}"
    )

    result = compute_params(
        cfg,
        attention=attention,
        ffn=ffn,
        tie_embeddings=args.tie_embeddings,
        num_dense_layers=num_dense,
        qk_norm=qk_norm,
    )

    if args.export:
        export_result(result, cfg, args, attention, ffn)

    print()
    print("| Component | Params | B | Ratio |")
    print("|---|---:|---:|---:|")
    for key in ["embedding", "attention", "layernorm", "ffn"]:
        val = result[key]
        print(
            f"| {key} | {val:,} | {val / 1e9:>8.4f} | {val / result['total_params'] * 100:>7.2f}% |"
        )
    if result["ffn"] != result["ffn_activated"]:
        for key in ["ffn_activated", "activated_total"]:
            val = result[key]
            print(
                f"| {key} | {val:,} | {val / 1e9:>8.4f} | {val / result['activated_total'] * 100:>7.2f}% |"
            )
    t = result["total_params"]
    print(f"| total_params | {t:,} | {t / 1e9:>8.4f} | {100:>7.2f}% |")
    print()


if __name__ == "__main__":
    main()
