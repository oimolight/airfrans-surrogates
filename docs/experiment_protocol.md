# AirfRANS experiment protocol

## Scope

The repository contains two completed experiments: a fixed-budget MLP/GNN/FNO comparison and a GNN relative-edge-feature ablation. No numbered intermediate experiment is planned.

## Shared conditions

| Item | Frozen value |
| --- | --- |
| Split | 160 train / 40 validation / 50 distribution-ID test cases |
| Seeds | 17, 29, 41 |
| Epochs | 100 |
| Optimiser steps | 16,000 per run |
| Validation interval | Every 10 epochs |
| Evaluation points | 2,048 per case, including 512 surface points |
| Device | CPU, eight PyTorch threads |
| Selection metric | Validation case-macro nondimensional pressure RMSE |
| Tie break | Earlier epoch |
| Test use | Exactly once per selected, locked checkpoint |

The reusable configurations are `configs/airfrans/model_comparison.json` and `configs/airfrans/edge_feature_ablation.json`. Exact historical configurations referenced by completed hashes are archived under `reports/provenance/2026-09-14/configs/airfrans/`. Every run stores its resolved configuration and hashes in `reports/runs/{run_id}/`.

## Model comparison

The MLP predicts each point independently. The GNN exchanges messages over geometry-derived edges. The FNO samples each case onto a regular grid and processes selected Fourier modes.

Cases, evaluation points, schedule, seeds, selection rule and metrics are fixed. Parameter count, graph construction, grid representation and receptive field differ, so this is not an architecture-only causal comparison.

Selected checkpoints are recorded in `reports/summaries/model_comparison_checkpoint_lock.json`; aggregate evidence is in `reports/summaries/model_comparison_evidence.md`.

## Edge-feature ablation

The ablation passes zero positions into GNN message passing, making every relative-position edge feature zero. It preserves node coordinates in node features, signed distance, surface indicator, inlet conditions, graph connectivity, model size and training budget.

It is therefore an edge-feature ablation, not removal of all geometry. Evidence is in `reports/summaries/edge_feature_ablation_checkpoint_lock.json` and `reports/summaries/edge_feature_ablation_report.md`.

## Evidence rules

- Failed runs remain in `reports/runs/`; they are not overwritten.
- Completed manifests, checkpoint hashes and test-attempt counts are immutable evidence.
- A low training loss is not evidence of generalisation.
- A seed-dependent difference is not a stable causal effect.
- Inference latency is not CFD solver runtime.
- Every reported number must resolve to a run ID or machine-readable summary.
