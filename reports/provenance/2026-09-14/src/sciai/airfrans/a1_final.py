from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sciai.common.artifacts import config_sha256, initialize_run, write_manifest

from .a1_contract import select_validation_checkpoint
from .a1_final_data import (
    FinalGridCase,
    FinalPointCase,
    evaluate_case,
    load_grid_case,
    load_point_case,
    normalize_grid_cases,
    normalize_point_cases,
    training_loss,
)
from .mlp_smoke import PointwiseMLP
from .pilot_models import PilotFNO, PilotGNN, trainable_scalar_count


EXPECTED_SELECTION_SHA256 = "8c532a15c2af6538177ae9885de8978553b162c4eb788d33a2998c9edc447014"
EXPERIMENT_ID = "airfrans-a1-final"
MODEL_NAMES = ("mlp", "gnn", "fno")
FIT_STATUS = "VALIDATION_SELECTED"
LOCK_SCHEMA_VERSION = 1
EVALUATOR_SOURCE_FILES = (
    Path("src/sciai/airfrans/a1_final.py"),
    Path("src/sciai/airfrans/a1_final_data.py"),
    Path("src/sciai/airfrans/a1_contract.py"),
    Path("src/sciai/airfrans/pilot_models.py"),
    Path("src/sciai/airfrans/mlp_smoke.py"),
)


class FinalA1Timeout(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def process_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def timeout_handler(signum: int, frame: object) -> None:
    raise FinalA1Timeout("Final A1 run exceeded its frozen wall-time limit")


def validate_protocol(base_config: Mapping[str, Any], selection_path: Path) -> dict[str, list[str]]:
    if file_sha256(selection_path) != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("Frozen selection file changed")
    if base_config["split"]["selection_file_sha256"] != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("Final config does not name the frozen selection hash")
    if base_config["seeds"] != [17, 29, 41]:
        raise RuntimeError("Final A1 seeds changed")
    training = base_config["training"]
    if training["epochs"] != 100 or training["validation_interval_epochs"] != 10:
        raise RuntimeError("Final A1 schedule changed")
    selection = read_json(selection_path)
    splits = {
        "train": list(selection["splits"]["train"]),
        "validation": list(selection["splits"]["validation"]),
        "id_test": list(selection["splits"]["id_test"]),
    }
    expected_counts = {"train": 160, "validation": 40, "id_test": 50}
    if {name: len(case_ids) for name, case_ids in splits.items()} != expected_counts:
        raise RuntimeError("Frozen split counts changed")
    if any(len(case_ids) != len(set(case_ids)) for case_ids in splits.values()):
        raise RuntimeError("Frozen split contains duplicate case IDs")
    if set(splits["train"]) & set(splits["validation"]):
        raise RuntimeError("Train and validation overlap")
    if (set(splits["train"]) | set(splits["validation"])) & set(splits["id_test"]):
        raise RuntimeError("Training/validation and test overlap")
    return splits


def resolved_config(
    base_config: Mapping[str, Any],
    model_name: str,
    seed: int,
    splits: Mapping[str, list[str]],
) -> dict[str, Any]:
    return {
        "schema_version": base_config["schema_version"],
        "purpose": base_config["purpose"],
        "model_name": model_name,
        "model": dict(base_config["models"][model_name]),
        "seed": seed,
        "device": base_config["device"],
        "sampling": dict(base_config["sampling"]),
        "evaluation": dict(base_config["evaluation"]),
        "training": dict(base_config["training"]),
        "split": {
            **base_config["split"],
            "train_case_ids": splits["train"],
            "validation_case_ids": splits["validation"],
            "test_case_ids_locked_after_all_fits": True,
        },
    }


def make_final_model(model_name: str, config: Mapping[str, Any]) -> nn.Module:
    model_config = config["model"]
    if model_name == "mlp":
        return PointwiseMLP(
            input_features=6,
            hidden_width=model_config["hidden_width"],
            hidden_layers=model_config["hidden_layers"],
        )
    if model_name == "gnn":
        return PilotGNN(
            input_features=6,
            hidden_width=model_config["hidden_width"],
            blocks=model_config["blocks"],
        )
    if model_name == "fno":
        return PilotFNO(
            input_features=6,
            width=model_config["width"],
            blocks=model_config["blocks"],
            modes=model_config["modes"],
        )
    raise ValueError(f"Unsupported final A1 model: {model_name}")


def aggregate_case_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("Cannot aggregate empty per-case metrics")
    return {
        "pressure_rmse_nondimensional_macro_mean": statistics.mean(
            float(record["pressure_rmse_nondimensional"]) for record in records
        ),
        "surface_mae_nondimensional_macro_mean": statistics.mean(
            float(record["surface_mae_nondimensional"]) for record in records
        ),
    }


def validate_case_records(
    records: Sequence[Mapping[str, Any]],
    split: str,
    expected_case_ids: Sequence[str],
) -> None:
    actual_case_ids = [record.get("case_id") for record in records]
    if len(records) != len(expected_case_ids):
        raise RuntimeError(
            f"Expected {len(expected_case_ids)} {split} records, got {len(records)}"
        )
    if any(record.get("split") != split for record in records):
        raise RuntimeError(f"Every per-case record must use split={split}")
    if len(actual_case_ids) != len(set(actual_case_ids)):
        raise RuntimeError(f"Duplicate case ID in {split} per-case records")
    if set(actual_case_ids) != set(expected_case_ids):
        raise RuntimeError(f"Per-case records do not match the frozen {split} case IDs")


def evaluator_source_hashes(repository_root: Path) -> dict[str, str]:
    return {
        path.as_posix(): file_sha256(repository_root / path)
        for path in EVALUATOR_SOURCE_FILES
    }


def _load_cases(
    model_name: str,
    dataset_root: Path,
    case_ids: Sequence[str],
    split: str,
    config: Mapping[str, Any],
) -> list[FinalPointCase | FinalGridCase]:
    if model_name == "fno":
        return [
            load_grid_case(
                dataset_root,
                case_id,
                split,
                config,
                include_evaluation=split != "train",
            )
            for case_id in case_ids
        ]
    return [
        load_point_case(dataset_root, case_id, split, config, model_name)
        for case_id in case_ids
    ]


def _normalise_cases(
    model_name: str,
    train_cases: list[FinalPointCase | FinalGridCase],
    evaluation_cases: list[FinalPointCase | FinalGridCase],
    config: Mapping[str, Any],
    normalizers: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if model_name == "fno":
        return normalize_grid_cases(
            train_cases,  # type: ignore[arg-type]
            evaluation_cases,  # type: ignore[arg-type]
            normalizers,
        )
    neighbors = config["model"]["neighbors"] if model_name == "gnn" else None
    return normalize_point_cases(
        train_cases,  # type: ignore[arg-type]
        evaluation_cases,  # type: ignore[arg-type]
        model_name,
        neighbors,
        normalizers,
    )


def _matching_runs(
    runs_root: Path,
    model_name: str,
    seed: int,
    expected_config_hash: str,
) -> list[tuple[Path, dict[str, Any]]]:
    matches = []
    for manifest_path in sorted(runs_root.glob(f"{EXPERIMENT_ID}-{model_name}-*/manifest.json")):
        manifest = read_json(manifest_path)
        if manifest.get("seed") == seed and manifest.get("config_sha256") == expected_config_hash:
            matches.append((manifest_path.parent, manifest))
    return matches


def fit_model(
    model_name: str,
    seed: int,
    config_path: Path,
    selection_path: Path,
    source_manifest_path: Path,
    dataset_root: Path,
    runs_root: Path,
    repository_root: Path,
    retry_failed: bool = False,
) -> Path:
    base_config = read_json(config_path)
    splits = validate_protocol(base_config, selection_path)
    if model_name not in MODEL_NAMES or seed not in base_config["seeds"]:
        raise ValueError("Model or seed is not part of the frozen A1 matrix")
    config = resolved_config(base_config, model_name, seed, splits)
    expected_config_hash = config_sha256(config)
    existing = _matching_runs(runs_root, model_name, seed, expected_config_hash)
    successful = [item for item in existing if item[1]["status"] in {FIT_STATUS, "COMPLETED"}]
    if successful:
        if len(successful) != 1:
            raise RuntimeError("More than one successful run exists for one frozen matrix cell")
        return successful[0][0]
    active = [
        item
        for item in existing
        if item[1]["status"] in {"CREATED", "RUNNING", "TEST_EVALUATING"}
    ]
    if active:
        raise RuntimeError("An equivalent run is already active; duplicate fit is disabled")
    if any(manifest["status"] == "FAILED" for _, manifest in existing) and not retry_failed:
        raise RuntimeError("A failed run already exists; automatic retry is disabled")

    run_id, run_directory = initialize_run(
        runs_root=runs_root,
        experiment_id=f"{EXPERIMENT_ID}-{model_name}",
        model=model_name,
        seed=seed,
        device=base_config["device"],
        source_airfrans_manifest=source_manifest_path,
        selected_case_manifest=selection_path,
        resolved_config=config,
        actual_case_counts={"train": 0, "validation": 0, "test": 0},
        repository_root=repository_root,
    )
    started_at = datetime.now(UTC).isoformat()
    overall_started = time.perf_counter()
    phase_started = overall_started
    timing: dict[str, Any] = {
        "started_at_utc": started_at,
        "preprocessing_seconds": None,
        "training_seconds": None,
        "validation_seconds": 0.0,
        "checkpoint_write_seconds": None,
        "fit_elapsed_seconds": None,
        "test_evaluation_seconds": None,
    }
    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    completed_steps = 0
    stop_reason = "FAILED_BEFORE_TRAINING"
    manifest = read_json(run_directory / "manifest.json")
    manifest.update(
        {
            "status": "RUNNING",
            "started_at_utc": started_at,
            "test_evaluation_attempts": 0,
            "test_results_used_for_selection": False,
            "stop_reason": None,
        }
    )
    write_manifest(run_directory, manifest)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(base_config["training"]["max_wall_seconds_per_run"])
    try:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(base_config["training"]["torch_threads"])
        torch.set_num_interop_threads(1)
        train_cases = _load_cases(
            model_name,
            dataset_root,
            splits["train"],
            "train",
            config,
        )
        validation_cases = _load_cases(
            model_name,
            dataset_root,
            splits["validation"],
            "validation",
            config,
        )
        normalizers = _normalise_cases(
            model_name,
            train_cases,
            validation_cases,
            config,
        )
        timing["preprocessing_seconds"] = time.perf_counter() - phase_started

        model = make_final_model(model_name, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
        generator = torch.Generator().manual_seed(seed)
        best_state: dict[str, Any] | None = None
        best_per_case: list[dict[str, Any]] | None = None
        training_seconds = 0.0
        for epoch in range(1, config["training"]["epochs"] + 1):
            epoch_started = time.perf_counter()
            model.train()
            epoch_losses = []
            for case_index in torch.randperm(len(train_cases), generator=generator).tolist():
                case = train_cases[case_index]
                optimizer.zero_grad(set_to_none=True)
                loss = training_loss(model_name, model, case)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite training loss at step {completed_steps + 1}")
                loss.backward()
                optimizer.step()
                completed_steps += 1
                epoch_losses.append(float(loss.detach()))
            epoch_seconds = time.perf_counter() - epoch_started
            training_seconds += epoch_seconds
            history.append(
                {
                    "epoch": epoch,
                    "steps_completed": completed_steps,
                    "training_normalized_mse_mean": statistics.mean(epoch_losses),
                    "training_seconds": epoch_seconds,
                }
            )
            if epoch % config["training"]["validation_interval_epochs"] == 0:
                validation_started = time.perf_counter()
                model.eval()
                with torch.no_grad():
                    per_case = [
                        evaluate_case(model_name, model, case, normalizers)
                        for case in validation_cases
                    ]
                validation_seconds = time.perf_counter() - validation_started
                timing["validation_seconds"] += validation_seconds
                aggregate = aggregate_case_metrics(per_case)
                validation_record = {
                    "epoch": epoch,
                    "validation_pressure_rmse_nondimensional": aggregate[
                        "pressure_rmse_nondimensional_macro_mean"
                    ],
                    "validation_surface_mae_nondimensional": aggregate[
                        "surface_mae_nondimensional_macro_mean"
                    ],
                    "validation_seconds": validation_seconds,
                }
                validation_history.append(validation_record)
                selected = select_validation_checkpoint(validation_history)
                if int(selected["epoch"]) == epoch:
                    best_state = {
                        "model_state": copy.deepcopy(model.state_dict()),
                        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                        "epoch": epoch,
                        "steps_completed": completed_steps,
                    }
                    best_per_case = per_case
            write_json(run_directory / "training_history.json", history)
            write_json(run_directory / "validation_history.json", validation_history)
        timing["training_seconds"] = training_seconds
        if best_state is None or best_per_case is None:
            raise RuntimeError("No validation checkpoint was selected")
        selected = select_validation_checkpoint(validation_history)
        if best_state["epoch"] != selected["epoch"]:
            raise RuntimeError("Saved best state differs from deterministic validation selection")
        checkpoint_started = time.perf_counter()
        checkpoint_path = run_directory / "checkpoints" / "selected.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model_name": model_name,
                "seed": seed,
                "config": config,
                "model_state": best_state["model_state"],
                "optimizer_state": best_state["optimizer_state"],
                "normalizers": normalizers,
                "selected_epoch": best_state["epoch"],
                "steps_at_selected_epoch": best_state["steps_completed"],
                "selection_metric": config["evaluation"]["checkpoint_metric"],
                "selection_metric_value": selected[
                    "validation_pressure_rmse_nondimensional"
                ],
            },
            checkpoint_path,
        )
        timing["checkpoint_write_seconds"] = time.perf_counter() - checkpoint_started
        timing["fit_elapsed_seconds"] = time.perf_counter() - overall_started
        timing["completed_at_utc"] = datetime.now(UTC).isoformat()
        stop_reason = "COMPLETED_100_EPOCHS_VALIDATION_SELECTED"
        checkpoint_hash = file_sha256(checkpoint_path)
        metrics = {
            "parameter_count": trainable_scalar_count(model),
            "epochs_completed": len(history),
            "optimizer_steps_completed": completed_steps,
            "selected_epoch": selected["epoch"],
            "selected_validation_pressure_rmse_nondimensional": selected[
                "validation_pressure_rmse_nondimensional"
            ],
            "selected_validation_surface_mae_nondimensional": selected[
                "validation_surface_mae_nondimensional"
            ],
            "checkpoint_selection_used_validation_only": True,
            "test_metrics": None,
            "test_set_evaluated": False,
            "peak_rss_bytes": process_peak_rss_bytes(),
            "stop_reason": stop_reason,
        }
        write_json(run_directory / "metrics.json", metrics)
        write_json(run_directory / "per_case_metrics.json", best_per_case)
        write_json(run_directory / "timing.json", timing)
        manifest = read_json(run_directory / "manifest.json")
        manifest.update(
            {
                "status": FIT_STATUS,
                "actual_case_counts": {"train": 160, "validation": 40, "test": 0},
                "checkpoint_path": checkpoint_path.resolve().relative_to(repository_root.resolve()).as_posix(),
                "checkpoint_sha256": checkpoint_hash,
                "selected_epoch": selected["epoch"],
                "selection_metric": config["evaluation"]["checkpoint_metric"],
                "selection_metric_value": selected[
                    "validation_pressure_rmse_nondimensional"
                ],
                "optimizer_steps_completed": completed_steps,
                "stop_reason": stop_reason,
                "fit_completed_at_utc": timing["completed_at_utc"],
                "peak_rss_bytes": metrics["peak_rss_bytes"],
            }
        )
        write_manifest(run_directory, manifest)
        print(json.dumps({"run_id": run_id, "run_directory": str(run_directory), "status": FIT_STATUS}))
        return run_directory
    except Exception as error:
        timing["fit_elapsed_seconds"] = time.perf_counter() - overall_started
        timing["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_json(run_directory / "training_history.json", history)
        write_json(run_directory / "validation_history.json", validation_history)
        write_json(run_directory / "timing.json", timing)
        manifest = read_json(run_directory / "manifest.json")
        manifest.update(
            {
                "status": "FAILED",
                "failure_reason": f"{type(error).__name__}: {error}",
                "optimizer_steps_completed": completed_steps,
                "stop_reason": f"FAILED_{type(error).__name__.upper()}",
                "failed_at_utc": timing["completed_at_utc"],
                "peak_rss_bytes": process_peak_rss_bytes(),
            }
        )
        write_manifest(run_directory, manifest)
        raise
    finally:
        signal.alarm(0)


def expected_selected_runs(
    config_path: Path,
    selection_path: Path,
    runs_root: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    base_config = read_json(config_path)
    splits = validate_protocol(base_config, selection_path)
    selected_runs = []
    for model_name in MODEL_NAMES:
        for seed in base_config["seeds"]:
            config = resolved_config(base_config, model_name, seed, splits)
            matches = _matching_runs(runs_root, model_name, seed, config_sha256(config))
            successful = [item for item in matches if item[1]["status"] in {FIT_STATUS, "COMPLETED"}]
            if len(successful) != 1:
                raise RuntimeError(
                    f"Expected one validation-selected run for {model_name} seed {seed}; found {len(successful)}"
                )
            selected_runs.append(successful[0])
    return selected_runs


def freeze_checkpoint_lock(
    config_path: Path,
    selection_path: Path,
    runs_root: Path,
    lock_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    selected_runs = expected_selected_runs(config_path, selection_path, runs_root)
    entries = []
    for run_directory, manifest in selected_runs:
        if manifest.get("test_evaluation_attempts") != 0:
            raise RuntimeError("Cannot create a pre-test lock after a test attempt")
        checkpoint_path = repository_root / manifest["checkpoint_path"]
        checkpoint_hash = file_sha256(checkpoint_path)
        if checkpoint_hash != manifest["checkpoint_sha256"]:
            raise RuntimeError("Selected checkpoint hash differs from its fit manifest")
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
        "schema_version": LOCK_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "test_results_available_when_created": False,
        "selection_manifest_sha256": EXPECTED_SELECTION_SHA256,
        "evaluation_source_sha256": evaluator_source_hashes(repository_root),
        "entries": entries,
    }
    if lock_path.exists():
        existing = read_json(lock_path)
        if existing.get("entries") != entries:
            raise RuntimeError("Existing checkpoint lock differs from selected fits")
        return existing
    write_json(lock_path, lock)
    return lock


def evaluate_locked_test_run(
    entry: Mapping[str, Any],
    lock_path: Path,
    config_path: Path,
    selection_path: Path,
    dataset_root: Path,
    repository_root: Path,
) -> Path:
    if not lock_path.exists():
        raise RuntimeError("Checkpoint lock must exist before test evaluation")
    lock = read_json(lock_path)
    if entry not in lock["entries"]:
        raise RuntimeError("Run is not present in the frozen checkpoint lock")
    if lock.get("evaluation_source_sha256") != evaluator_source_hashes(repository_root):
        raise RuntimeError("Frozen test evaluator source changed after checkpoint lock")
    run_directory = repository_root / entry["run_directory"]
    manifest = read_json(run_directory / "manifest.json")
    if manifest["status"] != FIT_STATUS or manifest.get("test_evaluation_attempts") != 0:
        raise RuntimeError("Test evaluation is allowed exactly once from VALIDATION_SELECTED")
    checkpoint_path = repository_root / entry["checkpoint_path"]
    if file_sha256(checkpoint_path) != entry["checkpoint_sha256"]:
        raise RuntimeError("Locked checkpoint changed before test evaluation")
    manifest.update(
        {
            "status": "TEST_EVALUATING",
            "test_evaluation_attempts": 1,
            "test_started_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    write_manifest(run_directory, manifest)
    started = time.perf_counter()
    try:
        base_config = read_json(config_path)
        splits = validate_protocol(base_config, selection_path)
        torch.set_num_threads(base_config["training"]["torch_threads"])
        torch.set_num_interop_threads(1)
        config = read_json(run_directory / "resolved_config.json")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config:
            raise RuntimeError("Checkpoint config differs from resolved run config")
        model_name = manifest["model"]
        model = make_final_model(model_name, config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        test_cases = _load_cases(
            model_name,
            dataset_root,
            splits["id_test"],
            "id_test",
            config,
        )
        _normalise_cases(
            model_name,
            [],
            test_cases,
            config,
            checkpoint["normalizers"],
        )
        with torch.no_grad():
            test_records = [
                evaluate_case(model_name, model, case, checkpoint["normalizers"])
                for case in test_cases
            ]
        aggregate = aggregate_case_metrics(test_records)
        per_case = read_json(run_directory / "per_case_metrics.json")
        validate_case_records(per_case, "validation", splits["validation"])
        validate_case_records(test_records, "id_test", splits["id_test"])
        if set(record["case_id"] for record in per_case) & set(
            record["case_id"] for record in test_records
        ):
            raise RuntimeError("Validation and test per-case records overlap")
        per_case.extend(test_records)
        metrics = read_json(run_directory / "metrics.json")
        fit_peak_rss_bytes = metrics["peak_rss_bytes"]
        test_peak_rss_bytes = process_peak_rss_bytes()
        metrics.update(
            {
                "test_pressure_rmse_nondimensional_macro_mean": aggregate[
                    "pressure_rmse_nondimensional_macro_mean"
                ],
                "test_surface_mae_nondimensional_macro_mean": aggregate[
                    "surface_mae_nondimensional_macro_mean"
                ],
                "test_set_evaluated": True,
                "test_evaluation_attempts": 1,
                "test_evaluated_checkpoint_sha256": entry["checkpoint_sha256"],
                "fit_peak_rss_bytes": fit_peak_rss_bytes,
                "test_peak_rss_bytes": test_peak_rss_bytes,
                "peak_rss_bytes": max(fit_peak_rss_bytes, test_peak_rss_bytes),
                "stop_reason": "COMPLETED_FROZEN_TEST_ONCE",
            }
        )
        timing = read_json(run_directory / "timing.json")
        timing["test_evaluation_seconds"] = time.perf_counter() - started
        timing["completed_at_utc"] = datetime.now(UTC).isoformat()
        write_json(run_directory / "per_case_metrics.json", per_case)
        write_json(run_directory / "metrics.json", metrics)
        write_json(run_directory / "timing.json", timing)
        manifest = read_json(run_directory / "manifest.json")
        manifest.update(
            {
                "status": "COMPLETED",
                "actual_case_counts": {"train": 160, "validation": 40, "test": 50},
                "test_evaluated_checkpoint_sha256": entry["checkpoint_sha256"],
                "test_results_used_for_selection": False,
                "test_completed_at_utc": timing["completed_at_utc"],
                "stop_reason": "COMPLETED_FROZEN_TEST_ONCE",
                "peak_rss_bytes": metrics["peak_rss_bytes"],
            }
        )
        write_manifest(run_directory, manifest)
        return run_directory
    except Exception as error:
        manifest = read_json(run_directory / "manifest.json")
        manifest.update(
            {
                "status": "FAILED",
                "failure_reason": f"{type(error).__name__}: {error}",
                "stop_reason": f"FAILED_TEST_{type(error).__name__.upper()}",
                "failed_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        write_manifest(run_directory, manifest)
        raise


def evaluate_all_locked_tests(
    lock_path: Path,
    config_path: Path,
    selection_path: Path,
    dataset_root: Path,
    repository_root: Path,
) -> None:
    lock = read_json(lock_path)
    for entry in lock["entries"]:
        command = [
            sys.executable,
            "-m",
            "sciai.airfrans.a1_final",
            "test-one",
            "--run-id",
            entry["run_id"],
            "--lock",
            str(lock_path.resolve()),
            "--config",
            str(config_path.resolve()),
            "--selection",
            str(selection_path.resolve()),
            "--dataset-root",
            str(dataset_root.resolve()),
            "--repository-root",
            str(repository_root.resolve()),
        ]
        completed = subprocess.run(command, cwd=repository_root, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Frozen test evaluation failed for {entry['run_id']} with exit code "
                f"{completed.returncode}"
            )


def evaluate_one_locked_test(
    run_id: str,
    lock_path: Path,
    config_path: Path,
    selection_path: Path,
    dataset_root: Path,
    repository_root: Path,
) -> Path:
    lock = read_json(lock_path)
    entries = [entry for entry in lock["entries"] if entry["run_id"] == run_id]
    if len(entries) != 1:
        raise RuntimeError(f"Expected one locked entry for run_id={run_id}")
    return evaluate_locked_test_run(
        entries[0],
        lock_path,
        config_path,
        selection_path,
        dataset_root,
        repository_root,
    )


def create_summary(
    config_path: Path,
    selection_path: Path,
    runs_root: Path,
    lock_path: Path,
    summary_json_path: Path,
    summary_markdown_path: Path,
    repository_root: Path,
) -> None:
    selected_runs = expected_selected_runs(config_path, selection_path, runs_root)
    if any(manifest["status"] != "COMPLETED" for _, manifest in selected_runs):
        raise RuntimeError("All nine locked test evaluations must complete before summary")
    splits = validate_protocol(read_json(config_path), selection_path)
    rows = []
    for run_directory, manifest in selected_runs:
        per_case = read_json(run_directory / "per_case_metrics.json")
        validation_records = [record for record in per_case if record.get("split") == "validation"]
        test_records = [record for record in per_case if record.get("split") == "id_test"]
        validate_case_records(validation_records, "validation", splits["validation"])
        validate_case_records(test_records, "id_test", splits["id_test"])
        if len(per_case) != 90:
            raise RuntimeError(f"Expected 90 per-case records in {run_directory.name}")
        metrics = read_json(run_directory / "metrics.json")
        timing = read_json(run_directory / "timing.json")
        rows.append(
            {
                "model": manifest["model"],
                "seed": manifest["seed"],
                "run_id": manifest["run_id"],
                "selected_epoch": metrics["selected_epoch"],
                "optimizer_steps_completed": metrics["optimizer_steps_completed"],
                "validation_pressure_rmse_nondimensional": metrics[
                    "selected_validation_pressure_rmse_nondimensional"
                ],
                "validation_surface_mae_nondimensional": metrics[
                    "selected_validation_surface_mae_nondimensional"
                ],
                "test_pressure_rmse_nondimensional": metrics[
                    "test_pressure_rmse_nondimensional_macro_mean"
                ],
                "test_surface_mae_nondimensional": metrics[
                    "test_surface_mae_nondimensional_macro_mean"
                ],
                "fit_seconds": timing["fit_elapsed_seconds"],
                "test_seconds": timing["test_evaluation_seconds"],
                "fit_peak_rss_bytes": metrics["fit_peak_rss_bytes"],
                "test_peak_rss_bytes": metrics["test_peak_rss_bytes"],
                "peak_rss_bytes": metrics["peak_rss_bytes"],
                "stop_reason": metrics["stop_reason"],
            }
        )
    failed_runs = []
    for manifest_path in sorted(runs_root.glob(f"{EXPERIMENT_ID}-*/manifest.json")):
        failed_manifest = read_json(manifest_path)
        if failed_manifest["status"] != "FAILED":
            continue
        timing_path = manifest_path.parent / "timing.json"
        failed_timing = read_json(timing_path) if timing_path.exists() else {}
        failed_runs.append(
            {
                "run_id": failed_manifest["run_id"],
                "model": failed_manifest["model"],
                "seed": failed_manifest["seed"],
                "failure_reason": failed_manifest["failure_reason"],
                "stop_reason": failed_manifest.get("stop_reason"),
                "optimizer_steps_completed": failed_manifest.get("optimizer_steps_completed"),
                "fit_elapsed_seconds": failed_timing.get("fit_elapsed_seconds"),
                "peak_rss_bytes": failed_manifest.get("peak_rss_bytes"),
                "test_evaluation_attempts": failed_manifest.get("test_evaluation_attempts", 0),
            }
        )
    model_summary = {}
    for model_name in MODEL_NAMES:
        model_rows = [row for row in rows if row["model"] == model_name]
        model_summary[model_name] = {
            "seeds": [row["seed"] for row in model_rows],
            "test_pressure_rmse_nondimensional_mean": statistics.mean(
                row["test_pressure_rmse_nondimensional"] for row in model_rows
            ),
            "test_pressure_rmse_nondimensional_std": statistics.stdev(
                row["test_pressure_rmse_nondimensional"] for row in model_rows
            ),
            "test_surface_mae_nondimensional_mean": statistics.mean(
                row["test_surface_mae_nondimensional"] for row in model_rows
            ),
            "test_surface_mae_nondimensional_std": statistics.stdev(
                row["test_surface_mae_nondimensional"] for row in model_rows
            ),
        }
    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "checkpoint_lock_path": lock_path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "selection_manifest_sha256": EXPECTED_SELECTION_SHA256,
        "test_evaluations_per_checkpoint": 1,
        "test_results_used_for_selection": False,
        "runs": rows,
        "failed_runs": failed_runs,
        "models": model_summary,
    }
    write_json(summary_json_path, summary)
    lines = [
        "# AirfRANS final A1 comparison",
        "",
        "Status: COMPLETED",
        "",
        "Checkpointは40 validation casesのcase-macro無次元pressure RMSEだけで選択した。",
        "全9 checkpointをlockした後、各checkpointについてfrozen 50-case distribution-ID testを1回だけ評価した。",
        "無次元誤差の基準量は各caseの `0.5 * |U_inlet|^2` であり、test正解から基準量を作っていない。",
        "GNNの近傍と翼横断判定はraw物理座標で作り、messageの相対変位はtrain統計で標準化した座標で表した。",
        "",
        "| Run ID | Model | Seed | Epoch | Steps | Val RMSE | Val surface MAE | Test RMSE | Test surface MAE | Fit s | Test s | Fit peak MiB | Test peak MiB | Stop reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run_id']}` | {row['model'].upper()} | {row['seed']} | {row['selected_epoch']} | "
            f"{row['optimizer_steps_completed']} | {row['validation_pressure_rmse_nondimensional']:.6g} | "
            f"{row['validation_surface_mae_nondimensional']:.6g} | {row['test_pressure_rmse_nondimensional']:.6g} | "
            f"{row['test_surface_mae_nondimensional']:.6g} | {row['fit_seconds']:.2f} | {row['test_seconds']:.2f} | "
            f"{row['fit_peak_rss_bytes'] / 2**20:.2f} | {row['test_peak_rss_bytes'] / 2**20:.2f} | "
            f"{row['stop_reason']} |"
        )
    lines.extend(["", "## Model aggregate", ""])
    for model_name in MODEL_NAMES:
        aggregate = model_summary[model_name]
        lines.append(
            f"- {model_name.upper()}: test RMSE {aggregate['test_pressure_rmse_nondimensional_mean']:.6g} "
            f"± {aggregate['test_pressure_rmse_nondimensional_std']:.6g}; surface MAE "
            f"{aggregate['test_surface_mae_nondimensional_mean']:.6g} ± "
            f"{aggregate['test_surface_mae_nondimensional_std']:.6g}."
        )
    lines.extend(["", "## Preserved failed runs", ""])
    if failed_runs:
        for failed in failed_runs:
            lines.append(
                f"- `{failed['run_id']}`: {failed['stop_reason']}; "
                f"{failed['failure_reason']}; steps={failed['optimizer_steps_completed']}; "
                f"test attempts={failed['test_evaluation_attempts']}."
            )
    else:
        lines.append("- 失敗runはなかった。")
    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "これはfrozen distribution-ID / unseen-exact-geometry split上の比較であり、geometry-OOD結果ではない。",
            "同一評価点とbudgetを使っても、grid、graph、受容野、parameter数が異なるためarchitecture単独の因果効果とは断定しない。",
            "A2/A3は実行していない。",
            "",
        ]
    )
    summary_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    summary_markdown_path.write_text("\n".join(lines), encoding="utf-8")


def run_all_subprocesses(args: argparse.Namespace) -> None:
    failures = []
    base_config = read_json(args.config)
    for model_name in MODEL_NAMES:
        for seed in base_config["seeds"]:
            command = [
                sys.executable,
                "-m",
                "sciai.airfrans.a1_final",
                "fit",
                "--model",
                model_name,
                "--seed",
                str(seed),
                "--config",
                str(args.config.resolve()),
                "--selection",
                str(args.selection.resolve()),
                "--source-manifest",
                str(args.source_manifest.resolve()),
                "--dataset-root",
                str(args.dataset_root.resolve()),
                "--runs-root",
                str(args.runs_root.resolve()),
                "--repository-root",
                str(args.repository_root.resolve()),
            ]
            completed = subprocess.run(command, cwd=args.repository_root, check=False)
            if completed.returncode != 0:
                failures.append({"model": model_name, "seed": seed, "returncode": completed.returncode})
    if failures:
        raise RuntimeError(f"Final A1 fits failed; test remains locked: {failures}")
    freeze_checkpoint_lock(
        args.config,
        args.selection,
        args.runs_root,
        args.lock,
        args.repository_root,
    )
    evaluate_all_locked_tests(
        args.lock,
        args.config,
        args.selection,
        args.dataset_root,
        args.repository_root,
    )
    create_summary(
        args.config,
        args.selection,
        args.runs_root,
        args.lock,
        args.summary_json,
        args.summary_markdown,
        args.repository_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "lock", "test", "test-one", "summary", "all"))
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--config", type=Path, default=Path("configs/airfrans/a1_final.json"))
    parser.add_argument("--selection", type=Path, default=Path("data/manifests/airfrans_selected_cases.json"))
    parser.add_argument("--source-manifest", type=Path, default=Path("data/airfrans_hf/data/Dataset/manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/airfrans_hf_selected/data/Dataset"))
    parser.add_argument("--runs-root", type=Path, default=Path("reports/runs"))
    parser.add_argument("--lock", type=Path, default=Path("reports/summaries/a1_checkpoint_lock.json"))
    parser.add_argument("--summary-json", type=Path, default=Path("reports/summaries/a1_final.json"))
    parser.add_argument("--summary-markdown", type=Path, default=Path("reports/summaries/a1_final.md"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "fit":
        if args.model is None or args.seed is None:
            raise ValueError("fit requires --model and --seed")
        fit_model(
            args.model,
            args.seed,
            args.config,
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
            args.selection,
            args.runs_root,
            args.lock,
            args.repository_root,
        )
    elif args.phase == "test":
        evaluate_all_locked_tests(
            args.lock,
            args.config,
            args.selection,
            args.dataset_root,
            args.repository_root,
        )
    elif args.phase == "test-one":
        if args.run_id is None:
            raise ValueError("test-one requires --run-id")
        evaluate_one_locked_test(
            args.run_id,
            args.lock,
            args.config,
            args.selection,
            args.dataset_root,
            args.repository_root,
        )
    elif args.phase == "summary":
        create_summary(
            args.config,
            args.selection,
            args.runs_root,
            args.lock,
            args.summary_json,
            args.summary_markdown,
            args.repository_root,
        )
    else:
        run_all_subprocesses(args)


if __name__ == "__main__":
    main()
