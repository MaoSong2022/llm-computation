# cs336 — Transformer Parameter Calculator

Compute the parameter count of any HuggingFace Transformer model directly from its `config.json` — no weights needed.

Supports **MHA**, **GQA**, **MQA**, **MLA** (Multi-head Latent Attention), **dense FFN**, **MoE**, and **hybrid** architectures (e.g. DeepSeek-V2 with dense + MoE layers).

## Installation

```bash
uv sync
```

Requires Python >= 3.10.

## Usage

```bash
uv run compute_model_params.py -m <model_id_or_config_path> [options]
```

### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--model` | `-m` | HuggingFace model ID (e.g. `deepseek-ai/DeepSeek-V2-Chat`) or local path to `config.json` |
| `--attention` | `-a` | Attention variant: `mha`, `gqa`, `mqa`, `mla` (auto-detected if omitted) |
| `--ffn` | `-f` | FFN variant: `silu`, `swiglu`, `gelu`, `relu`, `moe`, `moe_gated`, `deepseek_moe`. Use `+` for hybrid (e.g. `silu+deepseek_moe`) |
| `--num-dense-layers` | | Override number of initial dense layers (default: read from config) |
| `--qk-norm` / `--no-qk-norm` | | Enable/disable QK normalization (auto-detected from config if omitted) |
| `--tie-embeddings` / `--no-tie-embeddings` | `-t` | Override `tie_word_embeddings` (default: `false`) |

## Verification

Run these to verify correctness across supported architectures:

| Model | Attention | FFN | Command | Total | Verified |
|-------|-----------|-----|---------|-------|:--------:|
| DeepSeek-LLM 67B Chat | GQA | FFN | `-m deepseek-ai/deepseek-llm-67b-chat -a gqa` | 67.42B |yes|
| DeepSeek-MoE | GQA | DeepSeek MoE | `-m deepseek-ai/deepseek-moe-16b-chat -a mha -f deepseek_moe` | 16.87B | yes |
| DeepSeek-V2-Chat | MLA | DeepSeek MoE+FFN | `-m deepseek-ai/DeepSeek-V2-Chat -a mla -f silu+deepseek_moe --num-dense-layers 1` | 235.74B | yes |
| DeepSeek-V3 | MLA | DeepSeek MoE+FFN | `-m deepseek-ai/DeepSeek-V3 -a mla -f silu+deepseek_moe --num-dense-layers 3` | 671.02B | yes |
| DeepSeek-V4 Pro | CSA + HCA| DeepSeek MoE | TODO | TODO | |
| DeepSeek-V4 Flash | CSA + HCA| DeepSeek MoE | TODO | TODO | |
| Qwen3 8B | GQA | silu | `-m Qwen/Qwen3-8B -a gqa --qk-norm` | 8.19B | yes |
| Qwen3 235B-A22B | GQA | MoE | `-m Qwen/Qwen3-235B-A22B -a gqa -f moe --qk-norm` | 235.09B | yes |

> Note: Some models (e.g. Llama 3, Gemma 2) require authentication on HuggingFace. Use a local `config.json` or a token for access.

## Examples

**DeepSeek-V2-Chat** (MLA + DeepSeek MoE + 1 dense layer):

```bash
uv run compute_model_params.py -m deepseek-ai/DeepSeek-V2-Chat -a mla -f silu+deepseek_moe --num-dense-layers 1
```

**Llama / Mistral / Qwen** (auto-detect GQA + SiLU):

```bash
uv run compute_model_params.py -m meta-llama/Meta-Llama-3-8B
```

**Mixtral MoE** (auto-detect GQA + gated MoE):

```bash
uv run compute_model_params.py -m mistralai/Mixtral-8x7B-v0.1
```

**Local config file**:

```bash
uv run compute_model_params.py -m /path/to/config.json
```

## Supported Features

### Attention Variants

| Variant | Description |
|---------|-------------|
| `mha` | Multi-Head Attention (standard) |
| `gqa` | Grouped-Query Attention (e.g. Llama 2/3, Mistral) |
| `mqa` | Multi-Query Attention |
| `mla` | Multi-head Latent Attention with low-rank KV compression (DeepSeek-V2) |

### FFN Variants

| Variant | Description |
|---------|-------------|
| `silu` / `swiglu` | Gated FFN (3 linear projections) |
| `gelu` / `relu` | Non-gated FFN (2 linear projections) |
| `moe` | Standard MoE (1 expert linear layer + router) |
| `moe_gated` | Gated MoE (3 expert linear layers + router, e.g. Mixtral) |
| `deepseek_moe` | DeepSeek MoE (shared expert + routed experts, e.g. DeepSeek-V2) |

### Hybrid Models

Specify a dense FFN variant + an MoE variant separated by `+`, e.g. `silu+deepseek_moe`.
Use `--num-dense-layers` to set how many initial layers are dense; the rest are MoE.

## Output

For dense models:

```
Component                Params          B    Ratio
--------------------------------------------------
embedding            5,242,880,000    5.2429   30.92%
attention            1,697,185,280    1.6972   10.01%
layernorm                819,200    0.0008    0.00%
ffn                  7,019,868,160    7.0199   41.40%
--------------------------------------------------
total_params        14,534,272,000   14.5343  100.00%
```

For MoE models, additional rows show **activated** parameters (per-token expert usage):

```
ffn_activated         2,854,164,200    2.8542   20.84%
activated_total      10,368,824,040   10.3688  100.00%
```

## Project Structure

```
cs336/
├── compute_model_params.py     # CLI entry point
├── src/
│   └── param_compute/
│       ├── __init__.py          # compute_params() orchestration
│       ├── shared.py            # Shared param primitives (linear, embedding, layernorm)
│       ├── attention.py         # Attention param formulas (MHA, GQA, MQA, MLA)
│       ├── ffn.py               # Dense FFN param formulas
│       └── moe.py               # MoE param formulas (standard, gated, deepseek)
├── pyproject.toml
└── README.md
```
