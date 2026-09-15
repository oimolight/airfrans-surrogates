# Project status

Updated: 2026-09-15

## Completed

- Selected and audited the AirfRANS split: 160 train, 40 validation and 50 distribution-ID test cases.
- Implemented the point-data adapter, metrics, pointwise MLP, geometry-aware GNN and grid-based FNO.
- Completed the frozen MLP/GNN/FNO comparison for seeds 17/29/41.
- Selected checkpoints with validation data, locked checkpoint and evaluator hashes, then evaluated each selected checkpoint on test exactly once.
- Generated case-level comparison evidence, paired comparisons, representative validation fields and an inference profile.
- Re-rendered the committed model-comparison figures with Plotly and Kaleido from saved JSON and NPZ evidence; no model inference or test reevaluation was performed.
- Completed the preregistered relative-edge-feature ablation for all three seeds.
- Consolidated the workspace into the `airfrans-surrogates` uv package and removed the obsolete teaching scaffold.
- Declared Python 3.11.13 in `.tool-versions`; repository Python commands use `uv run --frozen`.
- Archived exact historical source/config bytes under `reports/provenance/2026-09-14/`, then adopted descriptive current module, config and lock filenames.
- Kept growing per-run outputs local while retaining compact aggregate JSON, reports and fixed-name figures as versioned evidence.
- Prepared an unpublished article draft with beginner-oriented explanations, equation images, evidence-linked result figures and a local rich-text copy helper. Human editorial and publication approval remain pending.

## Evidence

- Model-comparison summary: `reports/summaries/model_comparison_results.md`
- Model-comparison analysis: `reports/summaries/model_comparison_evidence.md`
- Historical model-comparison lock: `reports/summaries/model_comparison_checkpoint_lock.json`
- Edge-feature-ablation report: `reports/summaries/edge_feature_ablation_report.md`
- Historical edge-feature-ablation lock: `reports/summaries/edge_feature_ablation_checkpoint_lock.json`
- Local run provenance: `reports/runs/{run_id}/*.json`

## Repository validation

On 2026-09-15, the following offline checks passed after reorganisation:

- `uv lock --check`
- `uv sync --frozen --offline`
- `uv run --frozen --offline python -m pytest -q`: 53 pytest-native tests passed with no warnings reported after cleanup.
- `uv run --frozen --offline sciai-check`
- `uv run --frozen airfrans-edge-feature-ablation report` with temporary output paths reproduced the committed aggregate JSON from local run evidence.
- `--help` smoke checks for the seven documented AirfRANS entry points.
- SHA-256 comparison of all 19 evaluator-source entries listed across both checkpoint locks.

These checks did not download data, train a model or rerun frozen test evaluation. They validate repository structure and code paths; compact files under `reports/summaries/` retain the versioned scientific evidence, while full run manifests remain local.

## Not completed

- CFD solver timing or a CFD speed-up measurement.
- Total drag validation, production qualification or calibrated predictive uncertainty.
- Remote repository creation or Git push.

## Interpretation boundary

The evidence supports statements about the frozen distribution-ID protocol only. MLP and GNN errors overlap; FNO has higher error under this implementation and protocol. The edge-feature-ablation effect changes sign across seeds and does not remove all geometry information. These observations do not establish universal architecture rankings or causal mechanisms.
