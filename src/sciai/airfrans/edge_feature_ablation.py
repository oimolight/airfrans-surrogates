from __future__ import annotations

import argparse
import contextlib
import statistics
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sciai.common.artifacts import config_sha256

from . import model_comparison
from .models import GraphNeuralNetwork


EXPERIMENT_ID = "airfrans-edge-feature-ablation"
HISTORICAL_EXPERIMENT_ID = "airfrans-a3-edge-zero"
MODEL_NAME = "gnn"
ABLATION_FLAG = "relative_geometry_edge_features"
ABLATION_VALUE = "zeros"
SEEDS = (17, 29, 41)
SOURCE_FILE = Path("src/sciai/airfrans/edge_feature_ablation.py")
METRICS = (
    "pressure_rmse_nondimensional",
    "surface_mae_nondimensional",
)


class ZeroRelativeEdgeFeatureGNN(GraphNeuralNetwork):
    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(features, torch.zeros_like(positions), edge_index)


def make_ablation_model(model_name: str, config: Mapping[str, Any]) -> nn.Module:
    if model_name != MODEL_NAME:
        raise ValueError(f"Edge-feature ablation supports only {MODEL_NAME}, got {model_name}")
    model_config = config["model"]
    if model_config.get(ABLATION_FLAG) != ABLATION_VALUE:
        raise RuntimeError("Ablation config does not declare zero relative geometry edge features")
    return ZeroRelativeEdgeFeatureGNN(
        input_features=6,
        hidden_width=model_config["hidden_width"],
        blocks=model_config["blocks"],
    )


def evaluator_source_hashes(repository_root: Path) -> dict[str, str]:
    paths = (*model_comparison.EVALUATOR_SOURCE_FILES, SOURCE_FILE)
    return {path.as_posix(): model_comparison.file_sha256(repository_root / path) for path in paths}


@contextlib.contextmanager
def installed_ablation_runner() -> Iterator[None]:
    original_experiment_id = model_comparison.EXPERIMENT_ID
    original_model_names = model_comparison.MODEL_NAMES
    original_model_factory = model_comparison.make_model
    original_source_hashes = model_comparison.evaluator_source_hashes
    model_comparison.EXPERIMENT_ID = EXPERIMENT_ID
    model_comparison.MODEL_NAMES = (MODEL_NAME,)
    model_comparison.make_model = make_ablation_model
    model_comparison.evaluator_source_hashes = evaluator_source_hashes
    try:
        yield
    finally:
        model_comparison.EXPERIMENT_ID = original_experiment_id
        model_comparison.MODEL_NAMES = original_model_names
        model_comparison.make_model = original_model_factory
        model_comparison.evaluator_source_hashes = original_source_hashes


def validate_ablation_protocol(
    config_path: Path,
    baseline_config_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    config = model_comparison.read_json(config_path)
    baseline_config = model_comparison.read_json(baseline_config_path)
    model_comparison.validate_protocol(baseline_config, selection_path)
    model_comparison.validate_protocol(config, selection_path)
    for field in ("device", "sampling", "evaluation", "training", "split", "seeds"):
        if config[field] != baseline_config[field]:
            raise RuntimeError(f"Ablation changed frozen baseline field: {field}")
    if set(config["models"]) != {MODEL_NAME}:
        raise RuntimeError("Ablation config must contain only the GNN model")
    model_config = dict(config["models"][MODEL_NAME])
    if model_config.pop(ABLATION_FLAG, None) != ABLATION_VALUE:
        raise RuntimeError("Ablation must replace relative geometry edge features with zeros")
    if model_config != baseline_config["models"][MODEL_NAME]:
        raise RuntimeError("Ablation changed a frozen baseline GNN setting")
    if tuple(config["seeds"]) != SEEDS:
        raise RuntimeError("Ablation seeds differ from baseline GNN seeds")
    return config


def fit_seed(
    seed: int,
    config_path: Path,
    baseline_config_path: Path,
    selection_path: Path,
    source_manifest_path: Path,
    dataset_root: Path,
    runs_root: Path,
    repository_root: Path,
    retry_failed: bool,
) -> Path:
    config = validate_ablation_protocol(config_path, baseline_config_path, selection_path)
    if seed not in config["seeds"]:
        raise ValueError(f"Seed {seed} is not in the frozen ablation matrix")
    with installed_ablation_runner():
        return model_comparison.fit_model(
            MODEL_NAME,
            seed,
            config_path,
            selection_path,
            source_manifest_path,
            dataset_root,
            runs_root,
            repository_root,
            retry_failed,
        )


def expected_selected_runs(
    config_path: Path,
    baseline_config_path: Path,
    selection_path: Path,
    runs_root: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    validate_ablation_protocol(config_path, baseline_config_path, selection_path)
    with installed_ablation_runner():
        return model_comparison.expected_selected_runs(config_path, selection_path, runs_root)


def freeze_checkpoint_lock(
    config_path: Path,
    baseline_config_path: Path,
    selection_path: Path,
    runs_root: Path,
    lock_path: Path,
    baseline_lock_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    selected_runs = expected_selected_runs(
        config_path,
        baseline_config_path,
        selection_path,
        runs_root,
    )
    entries = []
    for run_directory, manifest in selected_runs:
        if manifest["status"] != model_comparison.FIT_STATUS or manifest.get("test_evaluation_attempts") != 0:
            raise RuntimeError("Ablation lock requires three validation-selected runs with no test attempts")
        checkpoint_path = repository_root / manifest["checkpoint_path"]
        checkpoint_hash = model_comparison.file_sha256(checkpoint_path)
        if checkpoint_hash != manifest["checkpoint_sha256"]:
            raise RuntimeError(f"Ablation checkpoint differs from manifest: {manifest['run_id']}")
        entries.append(
            {
                "model": manifest["model"],
                "seed": manifest["seed"],
                "run_id": manifest["run_id"],
                "run_directory": run_directory.resolve().relative_to(repository_root.resolve()).as_posix(),
                "config_sha256": manifest["config_sha256"],
                "checkpoint_path": manifest["checkpoint_path"],
                "checkpoint_sha256": checkpoint_hash,
                "selected_epoch": manifest["selected_epoch"],
                "selection_metric": manifest["selection_metric"],
                "selection_metric_value": manifest["selection_metric_value"],
            }
        )
    lock = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "ablation": {
            "name": "edge-feature ablation",
            "changed_feature": "relative geometry displacement on each directed edge",
            "replacement": ABLATION_VALUE,
            "node_features_retained": True,
            "graph_adjacency_retained": True,
        },
        "test_results_available_when_created": False,
        "selection_manifest_sha256": model_comparison.EXPECTED_SELECTION_SHA256,
        "config_path": config_path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "config_sha256": model_comparison.file_sha256(config_path),
        "baseline_config_sha256": model_comparison.file_sha256(baseline_config_path),
        "baseline_checkpoint_lock_path": baseline_lock_path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "baseline_checkpoint_lock_sha256": model_comparison.file_sha256(baseline_lock_path),
        "evaluation_source_sha256": evaluator_source_hashes(repository_root),
        "entries": entries,
    }
    if lock_path.exists():
        existing = model_comparison.read_json(lock_path)
        comparable_fields = set(lock) - {"created_at_utc"}
        if any(existing.get(field) != lock[field] for field in comparable_fields):
            raise RuntimeError("Existing ablation checkpoint lock differs from current selected fits")
        return existing
    model_comparison.write_json(lock_path, lock)
    return lock


def validate_checkpoint_lock(
    lock_path: Path,
    config_path: Path,
    baseline_config_path: Path,
    selection_path: Path,
    baseline_lock_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    validate_ablation_protocol(config_path, baseline_config_path, selection_path)
    lock = model_comparison.read_json(lock_path)
    experiment_id = lock.get("experiment_id")
    if experiment_id not in {EXPERIMENT_ID, HISTORICAL_EXPERIMENT_ID}:
        raise RuntimeError("Ablation checkpoint lock mismatch: experiment_id")
    if lock.get("selection_manifest_sha256") != model_comparison.EXPECTED_SELECTION_SHA256:
        raise RuntimeError("Ablation checkpoint lock mismatch: selection_manifest_sha256")

    historical = experiment_id == HISTORICAL_EXPERIMENT_ID
    config_candidates = [config_path]
    baseline_config_candidates = [baseline_config_path]
    if historical:
        archived_root = repository_root / model_comparison.PROVENANCE_ROOT
        config_candidates.append(archived_root / "configs/airfrans/a3_gnn_edge_zero.json")
        baseline_config_candidates.append(archived_root / "configs/airfrans/a1_final.json")

    if not any(
        path.is_file() and model_comparison.file_sha256(path) == lock.get("config_sha256")
        for path in config_candidates
    ):
        raise RuntimeError("Ablation checkpoint lock mismatch: config_sha256")
    baseline_config_hash = lock.get(
        "a1_config_sha256" if historical else "baseline_config_sha256"
    )
    if not any(
        path.is_file() and model_comparison.file_sha256(path) == baseline_config_hash
        for path in baseline_config_candidates
    ):
        raise RuntimeError("Ablation checkpoint lock mismatch: baseline_config_sha256")
    baseline_lock_hash = lock.get(
        "a1_checkpoint_lock_sha256" if historical else "baseline_checkpoint_lock_sha256"
    )
    if model_comparison.file_sha256(baseline_lock_path) != baseline_lock_hash:
        raise RuntimeError("Ablation checkpoint lock mismatch: baseline_checkpoint_lock_sha256")
    if not model_comparison.locked_source_hashes_match(
        repository_root, lock.get("evaluation_source_sha256", {})
    ):
        raise RuntimeError("Ablation checkpoint lock mismatch: evaluation_source_sha256")
    entries = lock.get("entries", [])
    if len(entries) != len(SEEDS) or {entry["seed"] for entry in entries} != set(SEEDS):
        raise RuntimeError("Ablation checkpoint lock does not contain the three frozen seeds")
    for entry in entries:
        if entry["model"] != MODEL_NAME:
            raise RuntimeError("Ablation checkpoint lock contains a non-GNN model")
        if model_comparison.file_sha256(repository_root / entry["checkpoint_path"]) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"Locked ablation checkpoint changed: {entry['run_id']}")
    return lock


def evaluate_one_test(
    run_id: str,
    lock_path: Path,
    config_path: Path,
    baseline_config_path: Path,
    selection_path: Path,
    dataset_root: Path,
    baseline_lock_path: Path,
    repository_root: Path,
) -> Path:
    validate_checkpoint_lock(
        lock_path,
        config_path,
        baseline_config_path,
        selection_path,
        baseline_lock_path,
        repository_root,
    )
    with installed_ablation_runner():
        return model_comparison.evaluate_one_locked_test(
            run_id,
            lock_path,
            config_path,
            selection_path,
            dataset_root,
            repository_root,
        )


def evaluate_all_tests(
    lock_path: Path,
    config_path: Path,
    baseline_config_path: Path,
    selection_path: Path,
    dataset_root: Path,
    baseline_lock_path: Path,
    repository_root: Path,
) -> None:
    lock = validate_checkpoint_lock(
        lock_path,
        config_path,
        baseline_config_path,
        selection_path,
        baseline_lock_path,
        repository_root,
    )
    for entry in lock["entries"]:
        manifest = model_comparison.read_json(repository_root / entry["run_directory"] / "manifest.json")
        if manifest["status"] != model_comparison.FIT_STATUS or manifest.get("test_evaluation_attempts") != 0:
            raise RuntimeError("Ablation test requires every checkpoint to be untested and validation-selected")
    for entry in lock["entries"]:
        command = [
            sys.executable,
            "-m",
            "sciai.airfrans.edge_feature_ablation",
            "test-one",
            "--run-id",
            entry["run_id"],
            "--lock",
            str(lock_path.resolve()),
            "--config",
            str(config_path.resolve()),
            "--baseline-config",
            str(baseline_config_path.resolve()),
            "--selection",
            str(selection_path.resolve()),
            "--dataset-root",
            str(dataset_root.resolve()),
            "--baseline-lock",
            str(baseline_lock_path.resolve()),
            "--repository-root",
            str(repository_root.resolve()),
        ]
        completed = subprocess.run(command, cwd=repository_root, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Ablation test evaluation failed for {entry['run_id']} with exit code "
                f"{completed.returncode}"
            )


def records_by_case(
    records: Sequence[Mapping[str, Any]],
    split: str,
    metric: str,
) -> dict[str, float]:
    return {
        str(record["case_id"]): float(record[metric])
        for record in records
        if record["split"] == split
    }


def source_reference(row: Mapping[str, Any]) -> str:
    return (
        f"Baseline=`{row['baseline_run_id']}` (`{row['baseline_metrics_path']}`, "
        f"`{row['baseline_per_case_path']}`); ablation=`{row['ablation_run_id']}` "
        f"(`{row['ablation_metrics_path']}`, `{row['ablation_per_case_path']}`)"
    )


def display_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def create_report(
    lock_path: Path,
    config_path: Path,
    baseline_config_path: Path,
    selection_path: Path,
    baseline_lock_path: Path,
    summary_json_path: Path,
    report_path: Path,
    repository_root: Path,
) -> None:
    validate_ablation_protocol(config_path, baseline_config_path, selection_path)
    lock = validate_checkpoint_lock(
        lock_path,
        config_path,
        baseline_config_path,
        selection_path,
        baseline_lock_path,
        repository_root,
    )
    baseline_lock = model_comparison.read_json(baseline_lock_path)
    if not model_comparison.locked_source_hashes_match(
        repository_root, baseline_lock["evaluation_source_sha256"]
    ):
        raise RuntimeError("Baseline evaluator source differs from its checkpoint lock")
    baseline_entries = {
        entry["seed"]: entry for entry in baseline_lock["entries"] if entry["model"] == MODEL_NAME
    }
    if set(baseline_entries) != set(SEEDS):
        raise RuntimeError("Baseline lock does not contain the three GNN seeds")

    rows = []
    for entry in sorted(lock["entries"], key=lambda value: value["seed"]):
        seed = int(entry["seed"])
        ablation_run_directory = repository_root / entry["run_directory"]
        ablation_manifest = model_comparison.read_json(ablation_run_directory / "manifest.json")
        if ablation_manifest["status"] != "COMPLETED" or ablation_manifest["test_evaluation_attempts"] != 1:
            raise RuntimeError(f"Ablation run is not a completed one-time evaluation: {entry['run_id']}")
        baseline_entry = baseline_entries[seed]
        baseline_run_directory = repository_root / baseline_entry["run_directory"]
        baseline_manifest = model_comparison.read_json(baseline_run_directory / "manifest.json")
        if baseline_manifest["status"] != "COMPLETED" or baseline_manifest["test_evaluation_attempts"] != 1:
            raise RuntimeError(f"Baseline run is not completed: {baseline_entry['run_id']}")
        baseline_config = model_comparison.read_json(baseline_run_directory / "resolved_config.json")
        ablation_config = model_comparison.read_json(ablation_run_directory / "resolved_config.json")
        for field in ("seed", "device", "sampling", "evaluation", "training", "split"):
            if baseline_config[field] != ablation_config[field]:
                raise RuntimeError(f"Ablation run changed paired baseline field: {field}")
        ablation_model_config = dict(ablation_config["model"])
        if ablation_model_config.pop(ABLATION_FLAG, None) != ABLATION_VALUE:
            raise RuntimeError("Resolved config does not contain the edge-feature ablation")
        if ablation_model_config != baseline_config["model"]:
            raise RuntimeError("Resolved model config differs beyond the ablation flag")

        baseline_metrics = model_comparison.read_json(baseline_run_directory / "metrics.json")
        ablation_metrics = model_comparison.read_json(ablation_run_directory / "metrics.json")
        baseline_per_case = model_comparison.read_json(baseline_run_directory / "per_case_metrics.json")
        ablation_per_case = model_comparison.read_json(ablation_run_directory / "per_case_metrics.json")
        row: dict[str, Any] = {
            "seed": seed,
            "baseline_run_id": baseline_entry["run_id"],
            "ablation_run_id": entry["run_id"],
            "baseline_metrics_path": (baseline_run_directory / "metrics.json").relative_to(repository_root).as_posix(),
            "ablation_metrics_path": (ablation_run_directory / "metrics.json").relative_to(repository_root).as_posix(),
            "baseline_per_case_path": (baseline_run_directory / "per_case_metrics.json").relative_to(repository_root).as_posix(),
            "ablation_per_case_path": (ablation_run_directory / "per_case_metrics.json").relative_to(repository_root).as_posix(),
            "baseline_selected_epoch": baseline_metrics["selected_epoch"],
            "ablation_selected_epoch": ablation_metrics["selected_epoch"],
            "baseline_optimizer_steps": baseline_metrics["optimizer_steps_completed"],
            "ablation_optimizer_steps": ablation_metrics["optimizer_steps_completed"],
            "parameter_count": ablation_metrics["parameter_count"],
        }
        if baseline_metrics["parameter_count"] != ablation_metrics["parameter_count"]:
            raise RuntimeError("Ablation parameter count differs from the paired baseline GNN")
        for metric in METRICS:
            metric_key = (
                "test_pressure_rmse_nondimensional_macro_mean"
                if metric == "pressure_rmse_nondimensional"
                else "test_surface_mae_nondimensional_macro_mean"
            )
            baseline_value = float(baseline_metrics[metric_key])
            ablation_value = float(ablation_metrics[metric_key])
            baseline_cases = records_by_case(baseline_per_case, "id_test", metric)
            ablation_cases = records_by_case(ablation_per_case, "id_test", metric)
            if set(baseline_cases) != set(ablation_cases) or len(baseline_cases) != 50:
                raise RuntimeError("Baseline and ablation test case records are not exactly paired")
            row[metric] = {
                "baseline": baseline_value,
                "ablation": ablation_value,
                "difference_ablation_minus_baseline": ablation_value - baseline_value,
                "ablation_lower_case_count": sum(
                    ablation_cases[case_id] < baseline_cases[case_id] for case_id in baseline_cases
                ),
                "baseline_lower_case_count": sum(
                    baseline_cases[case_id] < ablation_cases[case_id] for case_id in baseline_cases
                ),
                "tie_case_count": sum(
                    baseline_cases[case_id] == ablation_cases[case_id] for case_id in baseline_cases
                ),
            }
        rows.append(row)

    aggregates = {}
    for metric in METRICS:
        baseline_values = [float(row[metric]["baseline"]) for row in rows]
        ablation_values = [float(row[metric]["ablation"]) for row in rows]
        differences = [float(row[metric]["difference_ablation_minus_baseline"]) for row in rows]
        aggregates[metric] = {
            "seed_count": len(SEEDS),
            "baseline_mean": statistics.mean(baseline_values),
            "ablation_mean": statistics.mean(ablation_values),
            "difference_mean": statistics.mean(differences),
            "difference_sample_standard_deviation": statistics.stdev(differences),
            "difference_minimum": min(differences),
            "difference_maximum": max(differences),
        }

    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "experiment_id": EXPERIMENT_ID,
        "description": "GNN edge-feature ablation: relative geometry edge features replaced by zeros",
        "test_results_used_for_selection": False,
        "test_evaluations_per_checkpoint": 1,
        "rows": rows,
        "aggregates": aggregates,
        "ablation_checkpoint_lock_path": display_path(lock_path, repository_root),
        "baseline_checkpoint_lock_path": display_path(baseline_lock_path, repository_root),
    }
    model_comparison.write_json(summary_json_path, summary)

    metric_labels = {
        "pressure_rmse_nondimensional": "Pressure RMSE",
        "surface_mae_nondimensional": "Surface MAE",
    }
    measured_rows = []
    for row in rows:
        for metric in METRICS:
            values = row[metric]
            measured_rows.append(
                f"| {row['seed']} | {metric_labels[metric]} | {values['baseline']:.6f} | "
                f"{values['ablation']:.6f} | {values['difference_ablation_minus_baseline']:+.6f} | "
                f"{values['ablation_lower_case_count']} / {values['baseline_lower_case_count']} / "
                f"{values['tie_case_count']} | {source_reference(row)} |"
            )
    sensitivity_rows = []
    all_sources = "; ".join(source_reference(row) for row in rows)
    for metric in METRICS:
        values = aggregates[metric]
        sensitivity_rows.append(
            f"| {metric_labels[metric]} | {values['baseline_mean']:.6f} | "
            f"{values['ablation_mean']:.6f} | {values['difference_mean']:+.6f} | "
            f"{values['difference_sample_standard_deviation']:.6f} | "
            f"[{values['difference_minimum']:+.6f}, {values['difference_maximum']:+.6f}] | "
            f"{all_sources}; `{display_path(summary_json_path, repository_root)}` |"
        )
    run_contract_rows = "\n".join(
        f"| {row['seed']} | {row['baseline_selected_epoch']} | {row['ablation_selected_epoch']} | "
        f"{row['baseline_optimizer_steps']} | {row['ablation_optimizer_steps']} | "
        f"{row['parameter_count']} | {source_reference(row)} |"
        for row in rows
    )
    report = f"""# AirfRANS GNN edge-feature ablation

Status: COMPLETED

## Ablation definition

This experiment is an **edge-feature ablation**.

The relative geometry displacement on each directed edge was replaced with a zero tensor.

Graph adjacency was precomputed using the same raw coordinates, airfoil-crossing exclusion, and k-nearest-neighbor rule as the model-comparison baseline, then passed unchanged to the model.

The `x`, `y`, `inlet_velocity_x`, `inlet_velocity_y`, `signed_distance`, and `geometry_surface` node features were retained.

Therefore, this ablation does not remove all geometry information.

## Fixed contract

The train, validation, and test splits; sampling; graph adjacency; GNN depth and width; Adam optimiser; learning rate; epoch budget; validation interval; and seeds matched the baseline configuration in a mechanical comparison.

Sources: `{display_path(baseline_config_path, repository_root)}`; `{display_path(config_path, repository_root)}`; `{display_path(selection_path, repository_root)}`; `{display_path(lock_path, repository_root)}`.

Checkpoints were selected using only the same validation case-macro nondimensional pressure RMSE as the baseline, with the earlier epoch chosen in a tie.

The test evaluation used the same per-case nondimensional pressure RMSE and surface MAE as the baseline.

Differences are defined as `edge-zero ablation - GNN baseline`; a positive value means that the ablation increased the error.

| Seed | Baseline selected epoch | Edge-zero selected epoch | Baseline steps | Edge-zero steps | Parameters | Sources |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
{run_contract_rows}

## Measured difference

MEASURED: Each row pairs the same seed, frozen test cases, and evaluation points.

Case counts are ordered as `edge-zero lower / baseline lower / tie`.

| Seed | Metric | Baseline | Edge zero | Difference | Case counts | Sources |
| ---: | --- | ---: | ---: | ---: | --- | --- |
{chr(10).join(measured_rows)}

## Uncertainty and seed sensitivity

MEASURED: The sample standard deviation and range of the differences describe seed sensitivity; they are not confidence intervals.

| Metric | Baseline mean | Edge-zero mean | Mean difference | Difference sample SD | Difference range | Sources |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(sensitivity_rows)}

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
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "lock", "test", "test-one", "report"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--config", type=Path, default=Path("configs/airfrans/edge_feature_ablation.json"))
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/airfrans/model_comparison.json"))
    parser.add_argument("--selection", type=Path, default=Path("data/manifests/airfrans_selected_cases.json"))
    parser.add_argument("--source-manifest", type=Path, default=Path("data/airfrans_hf/data/Dataset/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/airfrans_hf_selected/data/Dataset"))
    parser.add_argument("--runs-root", type=Path, default=Path("reports/runs"))
    parser.add_argument("--lock", type=Path, default=Path("reports/summaries/edge_feature_ablation_checkpoint_lock.json"))
    parser.add_argument("--baseline-lock", type=Path, default=Path("reports/summaries/model_comparison_checkpoint_lock.json"))
    parser.add_argument("--summary-json", type=Path, default=Path("reports/summaries/edge_feature_ablation_data.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/summaries/edge_feature_ablation_report.md"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_ablation_protocol(args.config, args.baseline_config, args.selection)
    if args.phase == "fit":
        if args.seed is None:
            raise ValueError("fit requires --seed")
        fit_seed(
            args.seed,
            args.config,
            args.baseline_config,
            args.selection,
            args.source_manifest,
            args.dataset_root,
            args.runs_root,
            args.repository_root,
            args.retry_failed,
        )
    elif args.phase == "lock":
        freeze_checkpoint_lock(
            args.config,
            args.baseline_config,
            args.selection,
            args.runs_root,
            args.lock,
            args.baseline_lock,
            args.repository_root,
        )
    elif args.phase == "test":
        evaluate_all_tests(
            args.lock,
            args.config,
            args.baseline_config,
            args.selection,
            args.dataset_root,
            args.baseline_lock,
            args.repository_root,
        )
    elif args.phase == "test-one":
        if args.run_id is None:
            raise ValueError("test-one requires --run-id")
        evaluate_one_test(
            args.run_id,
            args.lock,
            args.config,
            args.baseline_config,
            args.selection,
            args.dataset_root,
            args.baseline_lock,
            args.repository_root,
        )
    else:
        create_report(
            args.lock,
            args.config,
            args.baseline_config,
            args.selection,
            args.baseline_lock,
            args.summary_json,
            args.report,
            args.repository_root,
        )


if __name__ == "__main__":
    main()
