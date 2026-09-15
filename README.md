# airfrans-surrogates

A reproducible comparison of three neural surrogate models for AirfRANS pressure fields: a pointwise multilayer perceptron (MLP), a graph neural network (GNN), and a Fourier neural operator (FNO). A surrogate model approximates an expensive numerical simulation with a learned predictor.

The repository contains the experiment code, frozen inputs, compact numerical evidence, reports, and figures for a completed three-seed comparison and a GNN edge-feature ablation. Raw CFD data, model checkpoints, and growing per-run outputs remain local.

## Results at a glance

The table reports the mean over 50 distribution-ID test cases after taking the median across three seeds for each case. Errors are nondimensional and lower is better.

| Model | Pressure RMSE | Surface pressure MAE |
| --- | ---: | ---: |
| MLP | 0.698543 | 0.551723 |
| GNN | 0.691280 | 0.553071 |
| FNO | 0.996148 | 0.800271 |

MLP and GNN errors overlap under this protocol, so the evidence does not support a universal ranking between them. FNO has higher error in this implementation. Removing relative-geometry edge features from the GNN produced small differences whose signs changed across seeds, so the ablation does not establish a stable effect.

These findings apply only to the frozen distribution-ID split. They do not demonstrate geometry-out-of-distribution generalisation, CFD replacement, production readiness, total-drag accuracy, calibrated uncertainty, or CFD speed-up.

## Quick start

The project requires [uv](https://docs.astral.sh/uv/) and Python 3.11.13, declared in `.tool-versions`.

```bash
uv sync --frozen
uv run --frozen sciai-check
uv run --frozen python -m pytest -q
```

These checks use committed metadata and synthetic fixtures. They do not download AirfRANS, train a model, or reevaluate the frozen test split.

## Read the evidence

- [Model-comparison evidence](reports/summaries/model_comparison_evidence.md) contains aggregate metrics, paired case comparisons, representative fields, and latency measurements.
- [Edge-feature-ablation report](reports/summaries/edge_feature_ablation_report.md) defines the intervention and reports its seed sensitivity.
- [Data contract](docs/data_contract.md) defines cases, inputs, targets, the 160/40/50 split, sampling, and metrics.
- [Experiment protocol](docs/experiment_protocol.md) records what changed and what remained fixed.
- [Reproducibility guide](docs/REPRODUCIBILITY.md) provides the full data, training, evaluation, and reporting commands.
- [Project status](docs/PROGRESS.md) separates completed work from open limitations.

## Reproduce the data setup

Inspect the pinned 751-file download plan before acquiring approximately 3.75 GB of selected AirfRANS data:

```bash
uv run --frozen python -m sciai.airfrans.hf_preflight
```

The actual network download is explicit:

```bash
uv run --frozen python -m sciai.airfrans.hf_download --execute
uv run --frozen python -m sciai.airfrans.one_case_audit
```

Full model training is CPU-intensive. Follow [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) rather than rerunning the completed test commands against existing local runs.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/sciai/airfrans/` | Data adapters, models, experiment runners, metrics, and evidence generation |
| `configs/airfrans/` | Frozen model-comparison and edge-feature-ablation inputs |
| `data/manifests/` | Frozen split and exact dataset download plan; no raw CFD fields |
| `tests/` | Focused synthetic and contract tests |
| `reports/runs/` | Local per-run provenance, metrics, checkpoints and predictions |
| `reports/summaries/` | Versioned aggregate evidence, reports, and checkpoint locks |
| `reports/figures/` | Versioned figures generated from saved evidence |
| `reports/provenance/` | Pre-Git source and config bytes required by historical hashes |

## Reproducibility boundary

`uv.lock`, the frozen configurations, the selected-case manifest, and compact result artifacts are versioned. `reports/runs/`, raw data, virtual environments, and model weights are intentionally excluded because they are large or grow with every experiment.

Historical run IDs and source-hash keys retain their original names inside immutable evidence. Current source, commands, and artifact names use descriptive terminology. The dated provenance snapshot exists because the completed experiments predate the initial Git commit; future experiments should record their Git commit instead of copying the source tree.
