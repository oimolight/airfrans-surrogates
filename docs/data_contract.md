# AirfRANS data contract

## Purpose

The task is supervised prediction of the AirfRANS pressure quantity `p/rho` around a two-dimensional airfoil. One case is one airfoil geometry under one inlet-flow condition.

## Case files

The implemented point-data path reads `{case_id}_internal.vtu` for mesh points, `implicit_distance` and pressure `p`. The GNN also reads `{case_id}_aerofoil.vtp` to filter edges that cross the airfoil.

The dataset itself is not committed. Its expected local root is `data/raw/airfrans_hf_selected/data/Dataset/`. The downloaded source manifest remains local at `data/airfrans_hf/data/Dataset/manifest.json`; the frozen selection and download-plan metadata are versioned under `data/manifests/`.

## Inputs and target

The six inputs at each point are $x$, $y$, two inlet-velocity components, signed distance and a geometry-derived surface indicator. The target is pressure `p`, interpreted by AirfRANS as `p/rho`. Ground-truth velocity is not an input, and sampling does not depend on pressure labels.

MLP and GNN consume sampled points. The GNN additionally receives a directed graph built from coordinates while excluding airfoil-crossing edges. FNO consumes a regular grid sampled from the same case and is queried at the common evaluation points.

## Frozen split

| Partition | Cases | Use |
| --- | ---: | --- |
| Train | 160 | Fit model parameters and training normalisers |
| Validation | 40 | Select checkpoints and diagnose implementation issues |
| Distribution-ID test | 50 | One-time evaluation of locked checkpoints |

The sets contain no repeated case IDs. The test split contains exact geometries not present in train or validation, but it is not a geometry-out-of-distribution benchmark.

The frozen selection SHA-256 is `8c532a15c2af6538177ae9885de8978553b162c4eb788d33a2998c9edc447014`.

## Sampling and metrics

Each evaluated case uses 2,048 common points: 512 surface points and 1,536 non-surface points. For prediction error $e=\hat{p}-p$, the nondimensional error is

$$
e^*=\frac{e}{0.5\lVert U_{inlet}\rVert^2}.
$$

The primary case-level metrics are pressure RMSE across all evaluation points and surface MAE across surface points. Aggregate metrics give each case equal weight.

## Leakage controls

- Checkpoint selection uses validation pressure RMSE only.
- Test labels are not used for fitting, normalisation, model selection or ablation definition.
- Label-dependent normalisers are fitted from permitted training targets.
- Model-comparison and edge-feature-ablation checkpoints are hash-locked before test evaluation.
- Existing completed test attempts must not be repeated or overwritten.
