# Run evidence

Each invocation creates a local directory named by a unique `run_id`.

Run directories contain records such as:

- `manifest.json`
- `resolved_config.json`
- `training_history.json`
- `metrics.json`
- `per_case_metrics.json`
- `timing.json`

These records, checkpoints and predictions remain local and are excluded from Git because the directory grows with every experiment. Unknown retrospective fields remain `null`; they are not reconstructed later.

Completed manifests are not overwritten. Publishable evidence is consolidated into the small machine-readable files under `reports/summaries/`; failed-run details needed by a conclusion must also be copied into a summary before publication.
