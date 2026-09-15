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

from . import a1_final
from .pilot_models import PilotGNN


EXPERIMENT_ID = "airfrans-a3-edge-zero"
MODEL_NAME = "gnn"
ABLATION_FLAG = "relative_geometry_edge_features"
ABLATION_VALUE = "zeros"
SEEDS = (17, 29, 41)
A3_SOURCE_FILE = Path("src/sciai/airfrans/a3_ablation.py")
METRICS = (
    "pressure_rmse_nondimensional",
    "surface_mae_nondimensional",
)


class ZeroRelativeEdgeFeatureGNN(PilotGNN):
    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(features, torch.zeros_like(positions), edge_index)


def make_ablation_model(model_name: str, config: Mapping[str, Any]) -> nn.Module:
    if model_name != MODEL_NAME:
        raise ValueError(f"A3 supports only {MODEL_NAME}, got {model_name}")
    model_config = config["model"]
    if model_config.get(ABLATION_FLAG) != ABLATION_VALUE:
        raise RuntimeError("A3 config does not declare zero relative geometry edge features")
    return ZeroRelativeEdgeFeatureGNN(
        input_features=6,
        hidden_width=model_config["hidden_width"],
        blocks=model_config["blocks"],
    )


def evaluator_source_hashes(repository_root: Path) -> dict[str, str]:
    paths = (*a1_final.EVALUATOR_SOURCE_FILES, A3_SOURCE_FILE)
    return {path.as_posix(): a1_final.file_sha256(repository_root / path) for path in paths}


@contextlib.contextmanager
def installed_ablation_runner() -> Iterator[None]:
    original_experiment_id = a1_final.EXPERIMENT_ID
    original_model_names = a1_final.MODEL_NAMES
    original_model_factory = a1_final.make_final_model
    original_source_hashes = a1_final.evaluator_source_hashes
    a1_final.EXPERIMENT_ID = EXPERIMENT_ID
    a1_final.MODEL_NAMES = (MODEL_NAME,)
    a1_final.make_final_model = make_ablation_model
    a1_final.evaluator_source_hashes = evaluator_source_hashes
    try:
        yield
    finally:
        a1_final.EXPERIMENT_ID = original_experiment_id
        a1_final.MODEL_NAMES = original_model_names
        a1_final.make_final_model = original_model_factory
        a1_final.evaluator_source_hashes = original_source_hashes


def validate_ablation_protocol(
    config_path: Path,
    a1_config_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    config = a1_final.read_json(config_path)
    a1_config = a1_final.read_json(a1_config_path)
    a1_final.validate_protocol(a1_config, selection_path)
    a1_final.validate_protocol(config, selection_path)
    for field in ("device", "sampling", "evaluation", "training", "split", "seeds"):
        if config[field] != a1_config[field]:
            raise RuntimeError(f"A3 changed frozen A1 field: {field}")
    if set(config["models"]) != {MODEL_NAME}:
        raise RuntimeError("A3 config must contain only the GNN model")
    model_config = dict(config["models"][MODEL_NAME])
    if model_config.pop(ABLATION_FLAG, None) != ABLATION_VALUE:
        raise RuntimeError("A3 must replace relative geometry edge features with zeros")
    if model_config != a1_config["models"][MODEL_NAME]:
        raise RuntimeError("A3 changed a frozen A1 GNN setting")
    if tuple(config["seeds"]) != SEEDS:
        raise RuntimeError("A3 seeds differ from A1 GNN seeds")
    return config


def fit_seed(
    seed: int,
    config_path: Path,
    a1_config_path: Path,
    selection_path: Path,
    source_manifest_path: Path,
    dataset_root: Path,
    runs_root: Path,
    repository_root: Path,
    retry_failed: bool,
) -> Path:
    config = validate_ablation_protocol(config_path, a1_config_path, selection_path)
    if seed not in config["seeds"]:
        raise ValueError(f"Seed {seed} is not in the frozen A3 matrix")
    with installed_ablation_runner():
        return a1_final.fit_model(
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
    a1_config_path: Path,
    selection_path: Path,
    runs_root: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    validate_ablation_protocol(config_path, a1_config_path, selection_path)
    with installed_ablation_runner():
        return a1_final.expected_selected_runs(config_path, selection_path, runs_root)


def freeze_checkpoint_lock(
    config_path: Path,
    a1_config_path: Path,
    selection_path: Path,
    runs_root: Path,
    lock_path: Path,
    a1_lock_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    selected_runs = expected_selected_runs(
        config_path,
        a1_config_path,
        selection_path,
        runs_root,
    )
    entries = []
    for run_directory, manifest in selected_runs:
        if manifest["status"] != a1_final.FIT_STATUS or manifest.get("test_evaluation_attempts") != 0:
            raise RuntimeError("A3 lock requires three validation-selected runs with no test attempts")
        checkpoint_path = repository_root / manifest["checkpoint_path"]
        checkpoint_hash = a1_final.file_sha256(checkpoint_path)
        if checkpoint_hash != manifest["checkpoint_sha256"]:
            raise RuntimeError(f"A3 checkpoint differs from manifest: {manifest['run_id']}")
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
        "selection_manifest_sha256": a1_final.EXPECTED_SELECTION_SHA256,
        "config_path": config_path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "config_sha256": a1_final.file_sha256(config_path),
        "a1_config_sha256": a1_final.file_sha256(a1_config_path),
        "a1_checkpoint_lock_path": a1_lock_path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "a1_checkpoint_lock_sha256": a1_final.file_sha256(a1_lock_path),
        "evaluation_source_sha256": evaluator_source_hashes(repository_root),
        "entries": entries,
    }
    if lock_path.exists():
        existing = a1_final.read_json(lock_path)
        comparable_fields = set(lock) - {"created_at_utc"}
        if any(existing.get(field) != lock[field] for field in comparable_fields):
            raise RuntimeError("Existing A3 checkpoint lock differs from current selected fits")
        return existing
    a1_final.write_json(lock_path, lock)
    return lock


def validate_checkpoint_lock(
    lock_path: Path,
    config_path: Path,
    a1_config_path: Path,
    selection_path: Path,
    a1_lock_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    validate_ablation_protocol(config_path, a1_config_path, selection_path)
    lock = a1_final.read_json(lock_path)
    expected_metadata = {
        "experiment_id": EXPERIMENT_ID,
        "selection_manifest_sha256": a1_final.EXPECTED_SELECTION_SHA256,
        "config_sha256": a1_final.file_sha256(config_path),
        "a1_config_sha256": a1_final.file_sha256(a1_config_path),
        "a1_checkpoint_lock_sha256": a1_final.file_sha256(a1_lock_path),
        "evaluation_source_sha256": evaluator_source_hashes(repository_root),
    }
    for field, expected in expected_metadata.items():
        if lock.get(field) != expected:
            raise RuntimeError(f"A3 checkpoint lock mismatch: {field}")
    entries = lock.get("entries", [])
    if len(entries) != len(SEEDS) or {entry["seed"] for entry in entries} != set(SEEDS):
        raise RuntimeError("A3 checkpoint lock does not contain the three frozen seeds")
    for entry in entries:
        if entry["model"] != MODEL_NAME:
            raise RuntimeError("A3 checkpoint lock contains a non-GNN model")
        if a1_final.file_sha256(repository_root / entry["checkpoint_path"]) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"Locked A3 checkpoint changed: {entry['run_id']}")
    return lock


def evaluate_one_test(
    run_id: str,
    lock_path: Path,
    config_path: Path,
    a1_config_path: Path,
    selection_path: Path,
    dataset_root: Path,
    a1_lock_path: Path,
    repository_root: Path,
) -> Path:
    validate_checkpoint_lock(
        lock_path,
        config_path,
        a1_config_path,
        selection_path,
        a1_lock_path,
        repository_root,
    )
    with installed_ablation_runner():
        return a1_final.evaluate_one_locked_test(
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
    a1_config_path: Path,
    selection_path: Path,
    dataset_root: Path,
    a1_lock_path: Path,
    repository_root: Path,
) -> None:
    lock = validate_checkpoint_lock(
        lock_path,
        config_path,
        a1_config_path,
        selection_path,
        a1_lock_path,
        repository_root,
    )
    for entry in lock["entries"]:
        manifest = a1_final.read_json(repository_root / entry["run_directory"] / "manifest.json")
        if manifest["status"] != a1_final.FIT_STATUS or manifest.get("test_evaluation_attempts") != 0:
            raise RuntimeError("A3 test requires every checkpoint to be untested and validation-selected")
    for entry in lock["entries"]:
        command = [
            sys.executable,
            "-m",
            "sciai.airfrans.a3_ablation",
            "test-one",
            "--run-id",
            entry["run_id"],
            "--lock",
            str(lock_path.resolve()),
            "--config",
            str(config_path.resolve()),
            "--a1-config",
            str(a1_config_path.resolve()),
            "--selection",
            str(selection_path.resolve()),
            "--dataset-root",
            str(dataset_root.resolve()),
            "--a1-lock",
            str(a1_lock_path.resolve()),
            "--repository-root",
            str(repository_root.resolve()),
        ]
        completed = subprocess.run(command, cwd=repository_root, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"A3 frozen test evaluation failed for {entry['run_id']} with exit code "
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
        f"A1=`{row['baseline_run_id']}` (`{row['baseline_metrics_path']}`, "
        f"`{row['baseline_per_case_path']}`); A3=`{row['ablation_run_id']}` "
        f"(`{row['ablation_metrics_path']}`, `{row['ablation_per_case_path']}`)"
    )


def create_report(
    lock_path: Path,
    config_path: Path,
    a1_config_path: Path,
    selection_path: Path,
    a1_lock_path: Path,
    summary_json_path: Path,
    report_path: Path,
    repository_root: Path,
) -> None:
    validate_ablation_protocol(config_path, a1_config_path, selection_path)
    lock = validate_checkpoint_lock(
        lock_path,
        config_path,
        a1_config_path,
        selection_path,
        a1_lock_path,
        repository_root,
    )
    a1_lock = a1_final.read_json(a1_lock_path)
    if a1_lock["evaluation_source_sha256"] != a1_final.evaluator_source_hashes(repository_root):
        raise RuntimeError("A1 evaluator source differs from its checkpoint lock")
    baseline_entries = {
        entry["seed"]: entry for entry in a1_lock["entries"] if entry["model"] == MODEL_NAME
    }
    if set(baseline_entries) != set(SEEDS):
        raise RuntimeError("A1 lock does not contain the three GNN baseline seeds")

    rows = []
    for entry in sorted(lock["entries"], key=lambda value: value["seed"]):
        seed = int(entry["seed"])
        ablation_run_directory = repository_root / entry["run_directory"]
        ablation_manifest = a1_final.read_json(ablation_run_directory / "manifest.json")
        if ablation_manifest["status"] != "COMPLETED" or ablation_manifest["test_evaluation_attempts"] != 1:
            raise RuntimeError(f"A3 run is not a completed one-time evaluation: {entry['run_id']}")
        baseline_entry = baseline_entries[seed]
        baseline_run_directory = repository_root / baseline_entry["run_directory"]
        baseline_manifest = a1_final.read_json(baseline_run_directory / "manifest.json")
        if baseline_manifest["status"] != "COMPLETED" or baseline_manifest["test_evaluation_attempts"] != 1:
            raise RuntimeError(f"A1 baseline run is not completed: {baseline_entry['run_id']}")
        baseline_config = a1_final.read_json(baseline_run_directory / "resolved_config.json")
        ablation_config = a1_final.read_json(ablation_run_directory / "resolved_config.json")
        for field in ("seed", "device", "sampling", "evaluation", "training", "split"):
            if baseline_config[field] != ablation_config[field]:
                raise RuntimeError(f"A3 run changed paired A1 field: {field}")
        ablation_model_config = dict(ablation_config["model"])
        if ablation_model_config.pop(ABLATION_FLAG, None) != ABLATION_VALUE:
            raise RuntimeError("A3 resolved config does not contain the edge-feature ablation")
        if ablation_model_config != baseline_config["model"]:
            raise RuntimeError("A3 resolved model config differs beyond the ablation flag")

        baseline_metrics = a1_final.read_json(baseline_run_directory / "metrics.json")
        ablation_metrics = a1_final.read_json(ablation_run_directory / "metrics.json")
        baseline_per_case = a1_final.read_json(baseline_run_directory / "per_case_metrics.json")
        ablation_per_case = a1_final.read_json(ablation_run_directory / "per_case_metrics.json")
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
            raise RuntimeError("A3 parameter count differs from the paired A1 GNN")
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
                raise RuntimeError("A1 and A3 test case records are not exactly paired")
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
        "a3_checkpoint_lock_path": lock_path.relative_to(repository_root).as_posix(),
        "a1_checkpoint_lock_path": a1_lock_path.relative_to(repository_root).as_posix(),
    }
    a1_final.write_json(summary_json_path, summary)

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
            f"{all_sources}; `{summary_json_path.relative_to(repository_root).as_posix()}` |"
        )
    run_contract_rows = "\n".join(
        f"| {row['seed']} | {row['baseline_selected_epoch']} | {row['ablation_selected_epoch']} | "
        f"{row['baseline_optimizer_steps']} | {row['ablation_optimizer_steps']} | "
        f"{row['parameter_count']} | {source_reference(row)} |"
        for row in rows
    )
    report = f"""# AirfRANS A3 GNN edge-feature ablation

Status: COMPLETED

## Ablation definition

この実験は**edge-feature ablation**である。

各directed edgeのrelative geometry displacementをzero tensorへ置き換えた。

graph adjacencyはA1と同じraw座標、翼横断除外、k-nearest-neighbor ruleで事前計算し、modelへそのまま渡した。

node featuresの `x`、`y`、`inlet_velocity_x`、`inlet_velocity_y`、`signed_distance`、`geometry_surface` は維持した。

したがって、このablationはすべてのgeometry informationを除去していない。

## Fixed contract

train、validation、test split、sampling、graph adjacency、GNN depth/width、Adam optimiser、learning rate、epoch budget、validation interval、seedはA1 configとの機械比較で一致した。

Sources: `{a1_config_path.relative_to(repository_root).as_posix()}`; `{config_path.relative_to(repository_root).as_posix()}`; `{selection_path.relative_to(repository_root).as_posix()}`; `{lock_path.relative_to(repository_root).as_posix()}`.

checkpointはA1と同じvalidation case-macro nondimensional pressure RMSEだけで選び、同値なら早いepochを選んだ。

testではA1と同じcase別nondimensional pressure RMSEとsurface MAEを使った。

差は `A3 ablation - A1 GNN baseline` と定義し、正値はablationの誤差増加を表す。

| Seed | A1 selected epoch | A3 selected epoch | A1 steps | A3 steps | Parameters | Sources |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
{run_contract_rows}

## Measured difference

MEASURED: 各行は同じseed、同じfrozen test cases、同じ評価点をpairにした。

case countsは `A3 lower / A1 lower / tie` の順である。

| Seed | Metric | A1 baseline | A3 edge-zero | Difference | Case counts | Sources |
| ---: | --- | ---: | ---: | ---: | --- | --- |
{chr(10).join(measured_rows)}

## Uncertainty and seed sensitivity

MEASURED: differenceのsample standard deviationとrangeはseed sensitivityの記述量であり、confidence intervalではない。

| Metric | A1 mean | A3 mean | Mean difference | Difference sample SD | Difference range | Sources |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(sensitivity_rows)}

## Interpretation

INTERPRETATION: measured differenceは、この固定A1 protocolでrelative geometry edge featureをzero化したときの変化である。

INTERPRETATION: seed別differenceの符号と幅が揃わない場合、平均差だけで安定した効果とは判断しない。

## What this ablation does not establish

このablationは、geometry information全体が必要かどうかを確かめる実験ではない。

node coordinates、signed distance、surface indicator、geometryから作ったadjacencyを残しているため、それらの寄与を分離しない。

relative geometry edge featureだけの変更でも、学習trajectoryとvalidation-selected epochは変わりうるため、個々のcheckpoint間の局所的な因果説明にはならない。

seed sensitivityは限られたseed集合の記述であり、母集団のuncertaintyや統計的有意性を確立しない。

distribution-ID splitだけを評価しているため、geometry-OOD性能を確立しない。

A1 test結果を見た後にhyperparameterやablation定義を変更しておらず、test結果はcheckpoint selectionに使っていない。

A2、追加ablation、記事草稿は実施していない。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "lock", "test", "test-one", "report"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--config", type=Path, default=Path("configs/airfrans/a3_gnn_edge_zero.json"))
    parser.add_argument("--a1-config", type=Path, default=Path("configs/airfrans/a1_final.json"))
    parser.add_argument("--selection", type=Path, default=Path("data/manifests/airfrans_selected_cases.json"))
    parser.add_argument("--source-manifest", type=Path, default=Path("data/airfrans_hf/data/Dataset/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/airfrans_hf_selected/data/Dataset"))
    parser.add_argument("--runs-root", type=Path, default=Path("reports/runs"))
    parser.add_argument("--lock", type=Path, default=Path("reports/summaries/a3_checkpoint_lock.json"))
    parser.add_argument("--a1-lock", type=Path, default=Path("reports/summaries/a1_checkpoint_lock.json"))
    parser.add_argument("--summary-json", type=Path, default=Path("reports/summaries/a3_ablation.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/summaries/a3_ablation.md"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_ablation_protocol(args.config, args.a1_config, args.selection)
    if args.phase == "fit":
        if args.seed is None:
            raise ValueError("fit requires --seed")
        fit_seed(
            args.seed,
            args.config,
            args.a1_config,
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
            args.a1_config,
            args.selection,
            args.runs_root,
            args.lock,
            args.a1_lock,
            args.repository_root,
        )
    elif args.phase == "test":
        evaluate_all_tests(
            args.lock,
            args.config,
            args.a1_config,
            args.selection,
            args.dataset_root,
            args.a1_lock,
            args.repository_root,
        )
    elif args.phase == "test-one":
        if args.run_id is None:
            raise ValueError("test-one requires --run-id")
        evaluate_one_test(
            args.run_id,
            args.lock,
            args.config,
            args.a1_config,
            args.selection,
            args.dataset_root,
            args.a1_lock,
            args.repository_root,
        )
    else:
        create_report(
            args.lock,
            args.config,
            args.a1_config,
            args.selection,
            args.a1_lock,
            args.summary_json,
            args.report,
            args.repository_root,
        )


if __name__ == "__main__":
    main()
