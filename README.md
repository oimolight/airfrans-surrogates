# airfrans-surrogates

This repository compares three neural surrogates for predicting AirfRANS pressure fields: a pointwise multilayer perceptron (MLP), a graph neural network (GNN), and a Fourier neural operator (FNO).

The completed evidence covers a frozen three-model comparison and a GNN edge-feature ablation. It does not establish CFD replacement, production readiness, total-drag accuracy, calibrated uncertainty, or generalisation to new geometry families.

## Start here

The project requires [uv](https://docs.astral.sh/uv/) for Python execution. Python 3.11.13 is declared in `.tool-versions`. From the repository root:

```bash
uv sync --frozen
uv run --frozen sciai-check
uv run --frozen python -m pytest
```

`uv.lock` is the dependency lock. Run every Python command through `uv run --frozen`; do not invoke bare `python`, `pytest`, or `pip` for repository work.

Repository checks and unit tests use synthetic fixtures or committed metadata. They do not download AirfRANS, retrain models, rerun frozen test evaluation, or validate the scientific findings by themselves.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/sciai/airfrans/` | Data adapters, models, runners, metrics and evidence generation |
| `configs/airfrans/` | Frozen model-comparison and edge-feature-ablation inputs |
| `data/manifests/` | Versioned dataset-selection metadata; no raw CFD fields |
| `tests/` | Synthetic and regression tests |
| `reports/runs/` | Local per-run provenance, metrics, checkpoints and predictions |
| `reports/summaries/` | Aggregate comparison and ablation evidence with checkpoint locks |
| `reports/figures/` | Figures referenced by the evidence reports |
| `reports/provenance/` | Exact historical source and config bytes referenced by completed locks |
| `docs/` | Data contract, protocol, reproducibility guide, sources and status |

## Reproduce or inspect the work

Read these in order:

1. [`docs/data_contract.md`](docs/data_contract.md) defines one case, inputs, targets, split and metrics.
2. [`docs/experiment_protocol.md`](docs/experiment_protocol.md) defines the comparison and ablation conditions.
3. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) gives uv, data and command-level instructions.
4. [`docs/PROGRESS.md`](docs/PROGRESS.md) records what actually ran and what did not.
5. [`reports/summaries/model_comparison_evidence.md`](reports/summaries/model_comparison_evidence.md) and [`reports/summaries/edge_feature_ablation_report.md`](reports/summaries/edge_feature_ablation_report.md) contain the measured comparison and ablation results.

The full dataset, virtual environment and model checkpoints are intentionally not versioned. The manifests and JSON evidence needed to audit the reported numbers are versioned.

## Expensive or irreversible commands

Dataset downloads, full training, frozen test evaluation and remote pushes require separate approval. Do not rerun `airfrans-model-comparison test` or `airfrans-edge-feature-ablation test` against the completed run directories: their manifests record the intended one-time test evaluations.

Historical run IDs and source-hash keys retain their original labels inside immutable evidence. Current source, configs, commands and artifact filenames use descriptive experiment names; exact historical source and configs are archived under `reports/provenance/2026-09-14/`.
