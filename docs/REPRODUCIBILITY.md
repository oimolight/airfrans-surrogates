# Reproducibility guide

## Environment

Install uv and run from the repository root:

```bash
uv sync --frozen
uv run --frozen python --version
uv run --frozen sciai-check
uv run --frozen python -m pytest
```

Python 3.11.13 is declared in `.tool-versions`. `uv.lock` is authoritative; `--frozen` prevents silent dependency-resolution changes. Run every repository Python command through `uv run --frozen`, not bare `python`, `pytest`, or `pip`. For an already prepared checkout without network access, add `--offline` to `uv sync` and `uv run`.

The recorded experiments ran on macOS 26.6.2 (arm64), Apple M2, 24 GiB memory, Python 3.11.13, uv 0.11.8 and PyTorch 2.14.0. Training used CPU with eight PyTorch threads. Run manifests record the device and software versions but not the CPU model, operating-system version or Git commit.

## Data preparation

The repository includes the frozen case manifest, not the CFD dataset. Dataset access requires separate network and storage approval.

Inspect the pinned Hugging Face download plan without downloading:

```bash
uv run --frozen python -m sciai.airfrans.hf_preflight
```

The actual download requires the explicit `--execute` flag:

```bash
uv run --frozen python -m sciai.airfrans.hf_download --execute
```

After approved acquisition, verify one case before training:

```bash
uv run --frozen python -m sciai.airfrans.one_case_audit
```

## Model comparison from scratch

Full training is long-running and requires explicit compute approval. Run each model with seeds 17, 29 and 41, for example:

```bash
uv run --frozen airfrans-model-comparison fit --model mlp --seed 17
```

Repeat for `mlp`, `gnn` and `fno` across all three seeds. Preserve failed runs and do not change the frozen data, model settings or budget to rescue a cell.

After all nine fits complete:

```bash
uv run --frozen airfrans-model-comparison lock
uv run --frozen airfrans-model-comparison test
uv run --frozen airfrans-model-comparison summary
```

The local manifests record one test attempt per completed checkpoint. An independent reproduction must use a new `--runs-root`, lock and summary path rather than overwrite the published aggregate evidence.

With local data and checkpoints, regenerate the comparison analysis with:

```bash
uv run --frozen airfrans-model-comparison-evidence
```

## Edge-feature ablation from scratch

With separate approval, fit the three ablation seeds, then lock, test and report:

```bash
uv run --frozen airfrans-edge-feature-ablation fit --seed 17
uv run --frozen airfrans-edge-feature-ablation fit --seed 29
uv run --frozen airfrans-edge-feature-ablation fit --seed 41
uv run --frozen airfrans-edge-feature-ablation lock
uv run --frozen airfrans-edge-feature-ablation test
uv run --frozen airfrans-edge-feature-ablation report
```

## Expected evidence

Per-run outputs are local working evidence and are excluded from Git because they grow with every experiment. Before publishing a result, consolidate the measurements needed to support its conclusions into the versioned files under `reports/summaries/`.

| Output | Meaning |
| --- | --- |
| `reports/runs/{run_id}/manifest.json` | Local run state, software, hashes and test-attempt count |
| `reports/runs/{run_id}/resolved_config.json` | Local exact resolved inputs |
| `reports/runs/{run_id}/per_case_metrics.json` | Local case-level records |
| `reports/summaries/model_comparison_checkpoint_lock.json` | Historical model-comparison checkpoint/evaluator hashes |
| `reports/summaries/model_comparison_evidence.md` | Model-comparison evidence and limitations |
| `reports/summaries/edge_feature_ablation_checkpoint_lock.json` | Historical edge-feature-ablation checkpoint/evaluator hashes |
| `reports/summaries/edge_feature_ablation_report.md` | Paired ablation result and seed sensitivity |

Raw fields, environments, model weights and per-run directories are intentionally local. Compact JSON evidence, summaries and final figures remain versioned.
