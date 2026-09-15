# AirfRANS model comparison

Status: COMPLETED

Checkpoints were selected using only case-macro nondimensional pressure RMSE on the 40 validation cases.
After all nine checkpoints were locked, each checkpoint was evaluated exactly once on the frozen 50-case distribution-ID test split.
The reference quantity for nondimensional error was `0.5 * |U_inlet|^2` for each case and was not derived from test labels.
GNN neighborhoods and airfoil-crossing checks used raw physical coordinates, while relative displacements in messages used coordinates standardized with training-set statistics.

| Run ID | Model | Seed | Epoch | Steps | Val RMSE | Val surface MAE | Test RMSE | Test surface MAE | Fit s | Test s | Fit peak MiB | Test peak MiB | Stop reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `airfrans-a1-final-mlp-20260914T161006605574Z-98daaecc748c` | MLP | 17 | 100 | 16000 | 0.512783 | 0.417312 | 0.724848 | 0.56482 | 53.98 | 9.36 | 443.38 | 375.83 | COMPLETED_FROZEN_TEST_ONCE |
| `airfrans-a1-final-mlp-20260914T161210862935Z-ebe9c7630884` | MLP | 29 | 80 | 16000 | 0.506982 | 0.417893 | 0.673539 | 0.528466 | 56.76 | 9.02 | 446.84 | 371.02 | COMPLETED_FROZEN_TEST_ONCE |
| `airfrans-a1-final-mlp-20260914T161320001837Z-213c345f12e8` | MLP | 41 | 90 | 16000 | 0.510367 | 0.424651 | 0.696636 | 0.554548 | 59.46 | 9.01 | 427.69 | 372.59 | COMPLETED_FROZEN_TEST_ONCE |
| `airfrans-a1-final-gnn-20260914T161630512662Z-e231245071ab` | GNN | 17 | 80 | 16000 | 0.498135 | 0.405285 | 0.692932 | 0.548811 | 988.35 | 71.71 | 430.56 | 493.56 | COMPLETED_FROZEN_TEST_ONCE |
| `airfrans-a1-final-gnn-20260914T163416373233Z-351ba431d17f` | GNN | 29 | 90 | 16000 | 0.500178 | 0.416283 | 0.6816 | 0.558251 | 1423.24 | 75.43 | 425.58 | 489.88 | COMPLETED_FROZEN_TEST_ONCE |
| `airfrans-a1-final-gnn-20260914T165934681307Z-11912419dc89` | GNN | 41 | 90 | 16000 | 0.512336 | 0.429996 | 0.72573 | 0.58272 | 1685.88 | 99.79 | 424.19 | 493.53 | COMPLETED_FROZEN_TEST_ONCE |
| `airfrans-a1-final-fno-20260914T174114930305Z-845b847d6a00` | FNO | 17 | 30 | 16000 | 0.706851 | 0.58962 | 0.993675 | 0.798104 | 500.61 | 18.28 | 572.73 | 432.19 | COMPLETED_FROZEN_TEST_ONCE |
| `airfrans-a1-final-fno-20260914T175311967526Z-4fd6c0d33d6e` | FNO | 29 | 70 | 16000 | 0.705528 | 0.583701 | 0.999522 | 0.800254 | 787.88 | 16.60 | 542.86 | 425.77 | COMPLETED_FROZEN_TEST_ONCE |
| `airfrans-a1-final-fno-20260914T180832172659Z-24f9c3e857b9` | FNO | 41 | 30 | 16000 | 0.699883 | 0.593707 | 0.996703 | 0.806756 | 643.89 | 18.31 | 636.45 | 427.28 | COMPLETED_FROZEN_TEST_ONCE |

## Model aggregate

- MLP: test RMSE 0.698341 ± 0.0256971; surface MAE 0.549278 ± 0.0187413.
- GNN: test RMSE 0.700087 ± 0.0229184; surface MAE 0.563261 ± 0.0175005.
- FNO: test RMSE 0.996633 ± 0.0029244; surface MAE 0.801704 ± 0.0045046.

## Preserved failed runs

- `airfrans-a1-final-fno-20260914T172917766147Z-dbc555a69904`: FAILED_KEYERROR; KeyError: 'models'; steps=0; test attempts=0.
- `airfrans-a1-final-fno-20260914T173040438405Z-83141265944c`: FAILED_VALUEERROR; ValueError: evaluation point has no valid FNO grid support; steps=1600; test attempts=0.
- `airfrans-a1-final-gnn-20260914T161441765247Z-7ca3029b4067`: FAILED_RUNTIMEERROR; RuntimeError: Node 1702 has only 11 non-crossing candidates; steps=0; test attempts=0.

## Limitation

This comparison uses a frozen distribution-ID / unseen-exact-geometry split and does not measure geometry-OOD performance.
Even with the same evaluation points and budget, the grids, graphs, receptive fields and parameter counts differ, so the results do not isolate a causal effect of architecture alone.
No geometry-OOD evaluation was performed.
