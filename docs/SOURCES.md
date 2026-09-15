# Sources and evidence boundaries

Reviewed: 2026-09-14

This repository uses external sources to explain the dataset and methodological background.
Experimental claims are supported by the run records in `reports/runs/` and the aggregates in `reports/summaries/`, not by external sources.

## AirfRANS

- [AirfRANS dataset documentation](https://airfrans.readthedocs.io/en/latest/notes/dataset.html): official description of dataset variables, tasks, and partitions.
- [PyTorch Geometric AirfRANS documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.AirfRANS.html): supplementary reference for the PyTorch Geometric data representation.

The case selection, inputs, target, sampling, normalization, and metrics used in this experiment are recorded in [`data_contract.md`](data_contract.md).
The repository's frozen contract takes precedence over general descriptions in external sources.

## Model background

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895): the original FNO paper.
- [NeuralOperator FNO theory guide](https://neuraloperator.github.io/dev/theory_guide/fno.html): supplementary explanation of Fourier layers.

The MLP, GNN, and FNO in this repository are specific implementations built for this comparison.
A paper title or model-family name alone does not imply an identical implementation or equivalent performance.

## Experimental evidence

- [`experiment_protocol.md`](experiment_protocol.md): conditions changed and held fixed in the model comparison and edge-feature ablation.
- [`../reports/summaries/model_comparison_evidence.md`](../reports/summaries/model_comparison_evidence.md): measured model-comparison results and run references.
- [`../reports/summaries/edge_feature_ablation_report.md`](../reports/summaries/edge_feature_ablation_report.md): measured edge-feature-ablation results and run references.

Repository checks and unit tests validate document structure and code paths on synthetic fixtures.
They do not remeasure trained-model performance.

## Operations not performed

This repository cleanup did not acquire data, train models, reevaluate the frozen test set, or push to a remote.
