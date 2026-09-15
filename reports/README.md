# Reports and evidence

| Path | Contents |
| --- | --- |
| `runs/` | Local per-run manifests, configurations, metrics, histories, checkpoints and predictions |
| `summaries/` | Cross-run evidence, checkpoint locks and profiles |
| `figures/` | Images referenced by evidence reports |
| `provenance/` | One pre-Git historical source/config snapshot referenced by the current checkpoint locks |
| top-level `airfrans_*` files | Download and split audit records retained for provenance |

Per-run outputs, raw data, virtual environments and model weights are local-only and excluded by `.gitignore`. Compact aggregate evidence, final reports and their fixed-name figures are committed. The representative-prediction NPZ under `summaries/` is the sole prediction exception because it supports the committed field figures.

Run IDs and original local artifact paths remain in the aggregate evidence for traceability. The dated provenance snapshot is retained because these experiments predate the initial Git commit; future experiments should record the source commit instead of copying source trees.
