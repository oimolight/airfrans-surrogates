# AirfRANS GNN edge-feature ablation

Status: COMPLETED

## Ablation definition

This experiment is an **edge-feature ablation**.

The relative geometry displacement on each directed edge was replaced with a zero tensor.

Graph adjacency was precomputed using the same raw coordinates, airfoil-crossing exclusion, and k-nearest-neighbor rule as the model-comparison baseline, then passed unchanged to the model.

The `x`, `y`, `inlet_velocity_x`, `inlet_velocity_y`, `signed_distance`, and `geometry_surface` node features were retained.

Therefore, this ablation does not remove all geometry information.

## Fixed contract

The train, validation, and test splits; sampling; graph adjacency; GNN depth and width; Adam optimiser; learning rate; epoch budget; validation interval; and seeds matched the baseline configuration in a mechanical comparison.

Sources: `reports/provenance/2026-09-14/configs/airfrans/a1_final.json`; `reports/provenance/2026-09-14/configs/airfrans/a3_gnn_edge_zero.json`; `data/manifests/airfrans_selected_cases.json`; `reports/summaries/edge_feature_ablation_checkpoint_lock.json`.

Checkpoints were selected using only the same validation case-macro nondimensional pressure RMSE as the baseline, with the earlier epoch chosen in a tie.

The test evaluation used the same per-case nondimensional pressure RMSE and surface MAE as the baseline.

Differences are defined as `edge-zero ablation - GNN baseline`; a positive value means that the ablation increased the error.

| Seed | Baseline selected epoch | Edge-zero selected epoch | Baseline steps | Edge-zero steps | Parameters | Sources |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 17 | 80 | 100 | 16000 | 16000 | 104513 | Baseline=`airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab` (`reports/runs/airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade/per_case_metrics.json`) |
| 29 | 90 | 90 | 16000 | 16000 | 104513 | Baseline=`airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f` (`reports/runs/airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc/per_case_metrics.json`) |
| 41 | 90 | 90 | 16000 | 16000 | 104513 | Baseline=`airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89` (`reports/runs/airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909/per_case_metrics.json`) |

## Measured difference

MEASURED: Each row pairs the same seed, frozen test cases, and evaluation points.

Case counts are ordered as `edge-zero lower / baseline lower / tie`.

| Seed | Metric | Baseline | Edge zero | Difference | Case counts | Sources |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 17 | Pressure RMSE | 0.692932 | 0.692140 | -0.000792 | 27 / 23 / 0 | Baseline=`airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab` (`reports/runs/airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade/per_case_metrics.json`) |
| 17 | Surface MAE | 0.548811 | 0.564579 | +0.015767 | 20 / 30 / 0 | Baseline=`airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab` (`reports/runs/airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade/per_case_metrics.json`) |
| 29 | Pressure RMSE | 0.681600 | 0.683610 | +0.002010 | 27 / 23 / 0 | Baseline=`airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f` (`reports/runs/airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc/per_case_metrics.json`) |
| 29 | Surface MAE | 0.558251 | 0.559127 | +0.000876 | 24 / 26 / 0 | Baseline=`airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f` (`reports/runs/airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc/per_case_metrics.json`) |
| 41 | Pressure RMSE | 0.725730 | 0.701509 | -0.024221 | 36 / 14 / 0 | Baseline=`airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89` (`reports/runs/airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909/per_case_metrics.json`) |
| 41 | Surface MAE | 0.582720 | 0.567027 | -0.015693 | 29 / 21 / 0 | Baseline=`airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89` (`reports/runs/airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909/per_case_metrics.json`) |

## Uncertainty and seed sensitivity

MEASURED: The sample standard deviation and range of the differences describe seed sensitivity; they are not confidence intervals.

| Metric | Baseline mean | Edge-zero mean | Mean difference | Difference sample SD | Difference range | Sources |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pressure RMSE | 0.700087 | 0.692420 | -0.007668 | 0.014404 | [-0.024221, +0.002010] | Baseline=`airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab` (`reports/runs/airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade/per_case_metrics.json`); Baseline=`airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f` (`reports/runs/airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc/per_case_metrics.json`); Baseline=`airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89` (`reports/runs/airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909/per_case_metrics.json`); `reports/summaries/edge_feature_ablation_data.json` |
| Surface MAE | 0.563261 | 0.563577 | +0.000317 | 0.015738 | [-0.015693, +0.015767] | Baseline=`airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab` (`reports/runs/airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T190425564535Z-3613f454bade/per_case_metrics.json`); Baseline=`airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f` (`reports/runs/airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T192438878011Z-bb72bfc321fc/per_case_metrics.json`); Baseline=`airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89` (`reports/runs/airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89/metrics.json`, `reports/runs/airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89/per_case_metrics.json`); ablation=`airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909` (`reports/runs/airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909/metrics.json`, `reports/runs/airfrans-a3-edge-zero-gnn-20260914T194601598835Z-94bdcd74f909/per_case_metrics.json`); `reports/summaries/edge_feature_ablation_data.json` |

## Interpretation

INTERPRETATION: The measured difference is the change observed when the relative-geometry edge feature was zeroed under this fixed model-comparison protocol.

INTERPRETATION: If the signs and magnitudes of the seed-level differences are inconsistent, the mean difference alone is not treated as a stable effect.

## What this ablation does not establish

This ablation does not test whether geometry information as a whole is necessary.

Because node coordinates, signed distance, the surface indicator, and geometry-derived adjacency remain, their contributions are not isolated.

Changing only the relative-geometry edge feature can still alter the training trajectory and validation-selected epoch, so the comparison does not provide a local causal explanation between individual checkpoints.

Seed sensitivity describes only the limited set of seeds and does not establish population uncertainty or statistical significance.

Because only the distribution-ID split was evaluated, the results do not establish geometry-OOD performance.

Hyperparameters and the ablation definition were not changed after observing baseline test results, and test results were not used for checkpoint selection.

Geometry-OOD evaluation and additional ablations were not performed.
