from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .a1_contract import bilinear_grid_to_points, case_pressure_metrics
from .a1_final import (
    MODEL_NAMES,
    _load_cases,
    _normalise_cases,
    evaluator_source_hashes,
    make_final_model,
)
from .a1_final_data import FinalGridCase, FinalPointCase


SEEDS = (17, 29, 41)
METRICS = (
    "pressure_rmse_nondimensional",
    "surface_mae_nondimensional",
)
PROFILE_REPEATS = {
    "preprocessing": 3,
    "model_forward": 30,
    "postprocessing_query": 30,
    "end_to_end": 3,
}
PROFILE_WARMUPS = {
    "preprocessing": 1,
    "model_forward": 5,
    "postprocessing_query": 5,
    "end_to_end": 1,
}


def distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9, method="linear")),
    }


def paired_case_summary(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> dict[str, Any]:
    if set(first) != set(second):
        raise ValueError("paired records must contain the same case IDs")
    differences = [float(first[case_id]) - float(second[case_id]) for case_id in sorted(first)]
    return {
        **distribution_summary(differences),
        "first_lower": sum(difference < 0 for difference in differences),
        "second_lower": sum(difference > 0 for difference in differences),
        "ties": sum(difference == 0 for difference in differences),
    }


def select_representative_cases(
    case_difficulty: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float | str]]:
    if not case_difficulty:
        raise ValueError("case_difficulty must not be empty")
    scores = {
        case_id: float(statistics.median(values))
        for case_id, values in case_difficulty.items()
    }
    population_median = float(statistics.median(scores.values()))
    median_case_id = min(scores, key=lambda case_id: (abs(scores[case_id] - population_median), case_id))
    high_case_id = min(scores, key=lambda case_id: (-scores[case_id], case_id))
    return {
        "median_difficulty": {
            "case_id": median_case_id,
            "difficulty_score": scores[median_case_id],
            "population_median": population_median,
        },
        "high_error": {
            "case_id": high_case_id,
            "difficulty_score": scores[high_case_id],
            "population_median": population_median,
        },
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def source_key(model_name: str, seed: int) -> str:
    return f"{model_name.upper()}{seed}"


def load_sources(repository_root: Path, lock_path: Path) -> dict[str, dict[str, Any]]:
    lock = read_json(lock_path)
    if lock["evaluation_source_sha256"] != evaluator_source_hashes(repository_root):
        raise RuntimeError("Locked evaluator source changed before A1 evidence analysis")
    sources: dict[str, dict[str, Any]] = {}
    for entry in lock["entries"]:
        run_directory = repository_root / entry["run_directory"]
        manifest = read_json(run_directory / "manifest.json")
        metrics = read_json(run_directory / "metrics.json")
        timing = read_json(run_directory / "timing.json")
        per_case = read_json(run_directory / "per_case_metrics.json")
        checkpoint_path = repository_root / entry["checkpoint_path"]
        if manifest["status"] != "COMPLETED" or manifest["test_evaluation_attempts"] != 1:
            raise RuntimeError(f"Run is not a completed one-time A1 evaluation: {entry['run_id']}")
        if manifest["test_results_used_for_selection"] is not False:
            raise RuntimeError(f"Run used test results for selection: {entry['run_id']}")
        if file_sha256(checkpoint_path) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed after lock: {entry['run_id']}")
        validation_records = [record for record in per_case if record["split"] == "validation"]
        test_records = [record for record in per_case if record["split"] == "id_test"]
        if len(validation_records) != 40 or len(test_records) != 50 or len(per_case) != 90:
            raise RuntimeError(f"Unexpected per-case evidence count: {entry['run_id']}")
        key = source_key(entry["model"], entry["seed"])
        sources[key] = {
            "model": entry["model"],
            "seed": entry["seed"],
            "run_id": entry["run_id"],
            "run_directory": entry["run_directory"],
            "checkpoint_path": entry["checkpoint_path"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "resolved_config_path": f"{entry['run_directory']}/resolved_config.json",
            "manifest_path": f"{entry['run_directory']}/manifest.json",
            "metrics_path": f"{entry['run_directory']}/metrics.json",
            "timing_path": f"{entry['run_directory']}/timing.json",
            "per_case_metrics_path": f"{entry['run_directory']}/per_case_metrics.json",
            "manifest": manifest,
            "metrics": metrics,
            "timing": timing,
            "validation_records": validation_records,
            "test_records": test_records,
        }
    expected = {source_key(model_name, seed) for model_name in MODEL_NAMES for seed in SEEDS}
    if set(sources) != expected:
        raise RuntimeError("Checkpoint lock does not contain the complete A1 model-seed matrix")
    return sources


def records_by_case(records: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    return {str(record["case_id"]): float(record[metric]) for record in records}


def build_aggregate_metrics(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model_name in MODEL_NAMES:
        model_keys = [source_key(model_name, seed) for seed in SEEDS]
        for metric in METRICS:
            values_by_seed = [records_by_case(sources[key]["test_records"], metric) for key in model_keys]
            case_ids = set(values_by_seed[0])
            if any(set(values) != case_ids for values in values_by_seed[1:]):
                raise RuntimeError("Test case IDs differ between seeds")
            case_seed_medians = [
                statistics.median(values[case_id] for values in values_by_seed)
                for case_id in sorted(case_ids)
            ]
            rows.append(
                {
                    "model": model_name,
                    "scope": "case_median_across_seeds",
                    "metric": metric,
                    **distribution_summary(case_seed_medians),
                    "source_keys": model_keys,
                }
            )
            for seed, key in zip(SEEDS, model_keys):
                values = [float(record[metric]) for record in sources[key]["test_records"]]
                rows.append(
                    {
                        "model": model_name,
                        "scope": f"seed_{seed}",
                        "metric": metric,
                        **distribution_summary(values),
                        "source_keys": [key],
                    }
                )
    return rows


def build_paired_comparisons(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for first_model, second_model in combinations(MODEL_NAMES, 2):
        first_keys = [source_key(first_model, seed) for seed in SEEDS]
        second_keys = [source_key(second_model, seed) for seed in SEEDS]
        for metric in METRICS:
            first_case_values: dict[str, list[float]] = {}
            second_case_values: dict[str, list[float]] = {}
            for first_key, second_key in zip(first_keys, second_keys):
                first_values = records_by_case(sources[first_key]["test_records"], metric)
                second_values = records_by_case(sources[second_key]["test_records"], metric)
                if set(first_values) != set(second_values):
                    raise RuntimeError("Paired test case IDs differ between models")
                for case_id in first_values:
                    first_case_values.setdefault(case_id, []).append(first_values[case_id])
                    second_case_values.setdefault(case_id, []).append(second_values[case_id])
                seed_summary = paired_case_summary(first_values, second_values)
                rows.append(
                    {
                        "first_model": first_model,
                        "second_model": second_model,
                        "scope": f"seed_{sources[first_key]['seed']}",
                        "metric": metric,
                        **seed_summary,
                        "source_keys": [first_key, second_key],
                    }
                )
            first_median = {
                case_id: statistics.median(values) for case_id, values in first_case_values.items()
            }
            second_median = {
                case_id: statistics.median(values) for case_id, values in second_case_values.items()
            }
            rows.append(
                {
                    "first_model": first_model,
                    "second_model": second_model,
                    "scope": "case_median_across_seeds",
                    "metric": metric,
                    **paired_case_summary(first_median, second_median),
                    "source_keys": [*first_keys, *second_keys],
                }
            )
    return rows


def representative_case_evidence(
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    case_difficulty: dict[str, list[float]] = {}
    all_keys = sorted(sources)
    for key in all_keys:
        for record in sources[key]["validation_records"]:
            case_difficulty.setdefault(str(record["case_id"]), []).append(
                float(record["pressure_rmse_nondimensional"])
            )
    if set(map(len, case_difficulty.values())) != {len(sources)}:
        raise RuntimeError("Validation cases are not common to all model-seed runs")
    selected = select_representative_cases(case_difficulty)
    return {
        role: {**details, "split": "validation", "source_keys": all_keys}
        for role, details in selected.items()
    }


def load_model_and_checkpoint(
    repository_root: Path,
    source: Mapping[str, Any],
) -> tuple[nn.Module, Mapping[str, Any], Mapping[str, Any]]:
    config = read_json(repository_root / source["resolved_config_path"])
    checkpoint = torch.load(
        repository_root / source["checkpoint_path"],
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint["config"] != config:
        raise RuntimeError(f"Checkpoint config differs from run config: {source['run_id']}")
    model = make_final_model(str(source["model"]), config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint, config


def load_preprocessed_case(
    model_name: str,
    case_id: str,
    dataset_root: Path,
    config: Mapping[str, Any],
    normalizers: Mapping[str, torch.Tensor],
) -> FinalPointCase | FinalGridCase:
    cases = _load_cases(model_name, dataset_root, [case_id], "validation", config)
    _normalise_cases(model_name, [], cases, config, normalizers)
    return cases[0]


def normalized_forward(
    model_name: str,
    model: nn.Module,
    case: FinalPointCase | FinalGridCase,
) -> torch.Tensor:
    if model_name == "mlp" and isinstance(case, FinalPointCase):
        return model(case.features)
    if model_name == "gnn" and isinstance(case, FinalPointCase) and case.edge_index is not None:
        return model(case.features, case.raw_positions, case.edge_index)
    if model_name == "fno" and isinstance(case, FinalGridCase):
        return model(case.features[None, ...])[0]
    raise TypeError(f"Unexpected case type for {model_name}")


def postprocess_prediction(
    model_name: str,
    normalized_output: torch.Tensor,
    case: FinalPointCase | FinalGridCase,
    normalizers: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(case, FinalPointCase):
        normalized_prediction = normalized_output
        evaluation = case.evaluation
    elif model_name == "fno" and isinstance(case, FinalGridCase) and case.evaluation is not None:
        normalized_prediction = bilinear_grid_to_points(
            normalized_output,
            case.evaluation.positions,
            case.bounds,
            valid=case.valid.squeeze(0),
        )
        evaluation = case.evaluation
    else:
        raise TypeError(f"Unexpected case type for {model_name}")
    prediction = normalized_prediction * normalizers["target_scale"] + normalizers["target_mean"]
    metrics = case_pressure_metrics(
        prediction,
        evaluation.target,
        evaluation.surface_mask,
        evaluation.inlet_velocity,
    )
    return prediction, metrics


def timed_samples(
    operation: Callable[[], object],
    repeats: int,
    warmups: int,
    collect_after_each: bool = False,
) -> list[float]:
    for _ in range(warmups):
        result = operation()
        del result
        if collect_after_each:
            gc.collect()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = operation()
        elapsed = time.perf_counter_ns() - started
        del result
        samples.append(elapsed / 1_000_000)
        if collect_after_each:
            gc.collect()
    return samples


def build_predictions_and_profile(
    repository_root: Path,
    dataset_root: Path,
    sources: Mapping[str, Mapping[str, Any]],
    representative_cases: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    predictions: dict[str, Any] = {}
    profile_rows = []
    for key in sorted(sources):
        source = sources[key]
        model_name = str(source["model"])
        model, checkpoint, config = load_model_and_checkpoint(repository_root, source)
        normalizers = checkpoint["normalizers"]
        for role, case_details in representative_cases.items():
            case_id = str(case_details["case_id"])
            case = load_preprocessed_case(model_name, case_id, dataset_root, config, normalizers)
            if case.evaluation is None:
                raise RuntimeError("Representative validation case has no evaluation points")
            with torch.no_grad():
                normalized_output = normalized_forward(model_name, model, case)
                prediction, _ = postprocess_prediction(model_name, normalized_output, case, normalizers)
            if role not in predictions:
                predictions[role] = {
                    "case_id": case_id,
                    "positions": case.evaluation.positions.detach().cpu().numpy(),
                    "surface_mask": case.evaluation.surface_mask.detach().cpu().numpy(),
                    "reference": case.evaluation.target.detach().cpu().numpy(),
                    "inlet_velocity": np.asarray(case.evaluation.inlet_velocity, dtype=np.float64),
                    "runs": {},
                }
            else:
                np.testing.assert_allclose(
                    predictions[role]["positions"],
                    case.evaluation.positions.detach().cpu().numpy(),
                    rtol=0.0,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    predictions[role]["reference"],
                    case.evaluation.target.detach().cpu().numpy(),
                    rtol=0.0,
                    atol=0.0,
                )
            predictions[role]["runs"][key] = prediction.detach().cpu().numpy()

            def preprocess() -> FinalPointCase | FinalGridCase:
                return load_preprocessed_case(model_name, case_id, dataset_root, config, normalizers)

            def forward() -> torch.Tensor:
                with torch.no_grad():
                    return normalized_forward(model_name, model, case)

            def postprocess() -> tuple[torch.Tensor, dict[str, float]]:
                with torch.no_grad():
                    return postprocess_prediction(model_name, normalized_output, case, normalizers)

            def end_to_end() -> tuple[torch.Tensor, dict[str, float]]:
                fresh_case = preprocess()
                with torch.no_grad():
                    output = normalized_forward(model_name, model, fresh_case)
                    return postprocess_prediction(model_name, output, fresh_case, normalizers)

            component_samples = {
                "preprocessing": timed_samples(
                    preprocess,
                    PROFILE_REPEATS["preprocessing"],
                    PROFILE_WARMUPS["preprocessing"],
                    collect_after_each=True,
                ),
                "model_forward": timed_samples(
                    forward,
                    PROFILE_REPEATS["model_forward"],
                    PROFILE_WARMUPS["model_forward"],
                ),
                "postprocessing_query": timed_samples(
                    postprocess,
                    PROFILE_REPEATS["postprocessing_query"],
                    PROFILE_WARMUPS["postprocessing_query"],
                ),
                "end_to_end": timed_samples(
                    end_to_end,
                    PROFILE_REPEATS["end_to_end"],
                    PROFILE_WARMUPS["end_to_end"],
                    collect_after_each=True,
                ),
            }
            profile_rows.append(
                {
                    "source_key": key,
                    "run_id": source["run_id"],
                    "model": model_name,
                    "seed": source["seed"],
                    "case_role": role,
                    "case_id": case_id,
                    "components_ms": {
                        component: {
                            "samples": samples,
                            "summary": distribution_summary(samples),
                        }
                        for component, samples in component_samples.items()
                    },
                }
            )
            del case, normalized_output, prediction
            gc.collect()
        del model, checkpoint
        gc.collect()
    profile = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "validation_representative_cases_only",
        "checkpoint_and_model_initialization_in_end_to_end": False,
        "filesystem_cache": "warmed_by_one_unmeasured_call_per_component",
        "torch_threads": 8,
        "platform": platform.platform(),
        "repeats": PROFILE_REPEATS,
        "warmups": PROFILE_WARMUPS,
        "rows": profile_rows,
    }
    return predictions, profile


def save_prediction_arrays(path: Path, predictions: Mapping[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for role, evidence in predictions.items():
        prefix = role
        arrays[f"{prefix}__case_id"] = np.asarray(evidence["case_id"])
        arrays[f"{prefix}__positions"] = evidence["positions"]
        arrays[f"{prefix}__surface_mask"] = evidence["surface_mask"]
        arrays[f"{prefix}__reference"] = evidence["reference"]
        arrays[f"{prefix}__inlet_velocity"] = evidence["inlet_velocity"]
        for key, prediction in evidence["runs"].items():
            arrays[f"{prefix}__{key}__prediction"] = prediction
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def pressure_norm(values: Sequence[np.ndarray]) -> Normalize:
    minimum = min(float(np.min(value)) for value in values)
    maximum = max(float(np.max(value)) for value in values)
    if minimum < 0 < maximum:
        return TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
    return Normalize(vmin=minimum, vmax=maximum)


def save_case_figure(
    path: Path,
    role: str,
    seed: int,
    evidence: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    prediction_artifact: str,
) -> None:
    keys = [source_key(model_name, seed) for model_name in MODEL_NAMES]
    positions = np.asarray(evidence["positions"])
    reference = np.asarray(evidence["reference"]).reshape(-1)
    inlet_velocity = np.asarray(evidence["inlet_velocity"])
    scale = 0.5 * float(np.dot(inlet_velocity, inlet_velocity))
    reference_scaled = reference / scale
    predictions_scaled = {
        key: np.asarray(evidence["runs"][key]).reshape(-1) / scale
        for key in keys
    }
    pressure_scale = pressure_norm([reference_scaled, *predictions_scaled.values()])
    errors = {key: np.abs(value - reference_scaled) for key, value in predictions_scaled.items()}
    error_maximum = max(float(np.max(value)) for value in errors.values())
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), sharex=True, sharey=True, constrained_layout=True)
    top_values = [("Reference p/rho divided by q", reference_scaled)] + [
        (f"{model_name.upper()} prediction", predictions_scaled[source_key(model_name, seed)])
        for model_name in MODEL_NAMES
    ]
    pressure_image = None
    for axis, (title, values) in zip(axes[0], top_values):
        pressure_image = axis.scatter(
            positions[:, 0], positions[:, 1], c=values, s=5, linewidths=0,
            cmap="RdBu_r", norm=pressure_scale, rasterized=True,
        )
        axis.set_title(title, fontsize=10)
    axes[1, 0].axis("off")
    run_lines = [f"{key}: {sources[key]['run_id']}" for key in keys]
    axes[1, 0].text(
        0.0,
        1.0,
        f"Rule: {role.replace('_', ' ')}\nSplit: validation\n\n" + "\n".join(run_lines),
        va="top",
        fontsize=7.5,
        wrap=True,
    )
    error_image = None
    for axis, model_name in zip(axes[1, 1:], MODEL_NAMES):
        key = source_key(model_name, seed)
        error_image = axis.scatter(
            positions[:, 0], positions[:, 1], c=errors[key], s=5, linewidths=0,
            cmap="magma", vmin=0.0, vmax=error_maximum, rasterized=True,
        )
        axis.set_title(f"{model_name.upper()} absolute error / q", fontsize=10)
    for axis in axes.flat:
        if axis.axison:
            axis.set_aspect("equal")
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]")
            axis.grid(color="#d7dde3", linewidth=0.35)
    if pressure_image is not None:
        figure.colorbar(pressure_image, ax=axes[0, :], shrink=0.78, label="dimensionless pressure")
    if error_image is not None:
        figure.colorbar(error_image, ax=axes[1, 1:], shrink=0.78, label="dimensionless absolute error")
    figure.suptitle(f"A1 {role.replace('_', ' ')} validation case, seed {seed}\n{evidence['case_id']}", fontsize=12)
    figure.text(
        0.5,
        0.005,
        f"Artifact: {prediction_artifact}",
        ha="center",
        fontsize=7,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def save_error_distribution_figure(
    path: Path,
    sources: Mapping[str, Mapping[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    labels = [model_name.upper() for model_name in MODEL_NAMES]
    colors = ["#287271", "#d67b32", "#805d93"]
    for axis, metric, title in zip(
        axes,
        METRICS,
        ("Pressure RMSE by frozen test case", "Surface MAE by frozen test case"),
    ):
        distributions = []
        for model_name in MODEL_NAMES:
            case_values: dict[str, list[float]] = {}
            for seed in SEEDS:
                key = source_key(model_name, seed)
                for record in sources[key]["test_records"]:
                    case_values.setdefault(str(record["case_id"]), []).append(float(record[metric]))
            distributions.append(np.asarray([statistics.median(values) for values in case_values.values()]))
        violins = axis.violinplot(distributions, showmeans=False, showmedians=True, showextrema=True)
        for body, color in zip(violins["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("#263238")
            body.set_alpha(0.72)
        for model_index, model_name in enumerate(MODEL_NAMES, start=1):
            seed_means = [
                statistics.mean(float(record[metric]) for record in sources[source_key(model_name, seed)]["test_records"])
                for seed in SEEDS
            ]
            axis.scatter(
                np.full(len(SEEDS), model_index) + np.linspace(-0.08, 0.08, len(SEEDS)),
                seed_means,
                color="#111111",
                s=20,
                zorder=3,
            )
        axis.set_xticks(range(1, len(labels) + 1), labels)
        axis.set_ylabel("dimensionless error")
        axis.set_title(title)
        axis.grid(axis="y", color="#d7dde3", linewidth=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def profile_run_summaries(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in profile["rows"]:
        grouped.setdefault(str(row["source_key"]), []).append(row)
    for key, case_rows in sorted(grouped.items()):
        for component in PROFILE_REPEATS:
            samples = [
                float(sample)
                for row in case_rows
                for sample in row["components_ms"][component]["samples"]
            ]
            rows.append(
                {
                    "source_key": key,
                    "run_id": case_rows[0]["run_id"],
                    "model": case_rows[0]["model"],
                    "seed": case_rows[0]["seed"],
                    "component": component,
                    **distribution_summary(samples),
                }
            )
    return rows


def build_accuracy_latency(
    sources: Mapping[str, Mapping[str, Any]],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_key: dict[str, list[float]] = {}
    for row in profile["rows"]:
        by_key.setdefault(str(row["source_key"]), []).append(
            float(row["components_ms"]["end_to_end"]["summary"]["median"])
        )
    return [
        {
            "source_key": key,
            "run_id": source["run_id"],
            "model": source["model"],
            "seed": source["seed"],
            "representative_case_end_to_end_latency_median_ms": float(statistics.median(by_key[key])),
            "test_pressure_rmse_nondimensional_macro_mean": float(
                source["metrics"]["test_pressure_rmse_nondimensional_macro_mean"]
            ),
            "test_surface_mae_nondimensional_macro_mean": float(
                source["metrics"]["test_surface_mae_nondimensional_macro_mean"]
            ),
        }
        for key, source in sorted(sources.items())
    ]


def save_accuracy_latency_figure(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    colors = {"mlp": "#287271", "gnn": "#d67b32", "fno": "#805d93"}
    label_offsets = {17: (5, 5), 29: (5, -13), 41: (5, 9)}
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("test_pressure_rmse_nondimensional_macro_mean", "test_surface_mae_nondimensional_macro_mean"),
        ("Test pressure RMSE versus inference latency", "Test surface MAE versus inference latency"),
    ):
        for row in rows:
            axis.scatter(
                row["representative_case_end_to_end_latency_median_ms"],
                row[metric],
                color=colors[str(row["model"])],
                s=52,
                label=str(row["model"]).upper(),
            )
            axis.annotate(
                str(row["seed"]),
                (row["representative_case_end_to_end_latency_median_ms"], row[metric]),
                xytext=label_offsets[int(row["seed"])],
                textcoords="offset points",
                fontsize=8,
            )
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), frameon=False)
        axis.set_xscale("log")
        axis.set_xlabel("validation representative-case end-to-end latency [ms, log scale]")
        axis.set_ylabel("dimensionless error; lower is better")
        axis.set_title(title)
        axis.grid(color="#d7dde3", linewidth=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def format_number(value: float) -> str:
    return f"{value:.6f}"


def sources_text(keys: Sequence[str], sources: Mapping[str, Mapping[str, Any]]) -> str:
    return ", ".join(f"{key}=`{sources[key]['run_id']}`" for key in keys)


def build_report(
    sources: Mapping[str, Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    representative_cases: Mapping[str, Mapping[str, Any]],
    profile_rows: Sequence[Mapping[str, Any]],
    figures: Mapping[str, Any],
    evidence_artifact: str,
    profile_artifact: str,
    prediction_artifact: str,
) -> str:
    all_keys = sorted(sources)
    source_rows = "\n".join(
        f"| {key} | `{source['run_id']}` | `{source['per_case_metrics_path']}`; "
        f"`{source['metrics_path']}`; `{source['timing_path']}`; `{source['checkpoint_path']}` |"
        for key, source in sorted(sources.items())
    )
    aggregate_rows = "\n".join(
        f"| {row['model'].upper()} | {row['scope']} | `{row['metric']}` | {row['count']} | "
        f"{format_number(row['mean'])} | {format_number(row['median'])} | {format_number(row['p90'])} | "
        f"{sources_text(row['source_keys'], sources)}; `{evidence_artifact}` |"
        for row in aggregates
    )
    paired_rows = "\n".join(
        f"| {row['first_model'].upper()} - {row['second_model'].upper()} | {row['scope']} | `{row['metric']}` | "
        f"{row['count']} | {format_number(row['mean'])} | {format_number(row['median'])} | "
        f"{format_number(row['p90'])} | {row['first_lower']} / {row['second_lower']} / {row['ties']} | "
        f"{sources_text(row['source_keys'], sources)}; `{evidence_artifact}` |"
        for row in paired
    )
    representative_rows = "\n".join(
        f"| {role} | `{details['case_id']}` | {format_number(details['difficulty_score'])} | "
        f"{format_number(details['population_median'])} | {sources_text(details['source_keys'], sources)}; "
        f"`{evidence_artifact}` |"
        for role, details in representative_cases.items()
    )
    profile_table_rows = "\n".join(
        f"| {row['model'].upper()} | {row['seed']} | {row['component']} | {row['count']} | "
        f"{format_number(row['mean'])} | {format_number(row['median'])} | {format_number(row['p90'])} | "
        f"{sources_text([row['source_key']], sources)}; `{profile_artifact}`; "
        f"`{sources[row['source_key']]['checkpoint_path']}` |"
        for row in profile_rows
    )
    case_figure_lines = []
    for role in ("median_difficulty", "high_error"):
        for seed in SEEDS:
            path = figures["cases"][role][str(seed)]
            keys = [source_key(model_name, seed) for model_name in MODEL_NAMES]
            case_figure_lines.extend(
                [
                    f"### {role}, seed {seed}",
                    "",
                    f"MEASURED: reference pressure、prediction、absolute errorを同じvalidation caseで示す。 "
                    f"Sources: {sources_text(keys, sources)}; `{prediction_artifact}`; `{path}`.",
                    "",
                    f"![{role} seed {seed}](../figures/a1_evidence/{Path(path).name})",
                    "",
                ]
            )
    return f"""# AirfRANS A1 evidence

Status: COMPLETED A1 ANALYSIS; NO RETRAINING OR TUNING; A3 AND ARTICLE DRAFTING NOT STARTED

## Evidence labels

**MEASURED**はcompleted run artifact、locked checkpointを使ったvalidation inference、またはこの分析で保存したprofile sampleから直接計算した値を示す。

**INTERPRETATION**は測定値から読める範囲の説明を示す。

**HYPOTHESIS**は追加実験なしでは確定できない説明候補を示す。

すべての数値表のSources列は次のsource registryにあるrun_idとartifact pathへ解決される。

## Source registry

| Key | run_id | Artifact paths |
| --- | --- | --- |
{source_rows}

分析artifactは `{evidence_artifact}`、profile artifactは `{profile_artifact}`、field artifactは `{prediction_artifact}` である。

## Aggregate test metrics

MEASURED: frozen matrixのseedは個別行で示し、case_median_across_seeds行は各共通caseでseed中央値を作ってから集約した。

p90はNumPyのlinear quantile、誤差は小さいほどよい。

| Model | Scope | Metric | Count | Mean | Median | P90 | Sources |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
{aggregate_rows}

## Paired per-case comparisons

MEASURED: 差はfirst model minus second modelである。

負の差はfirst modelの誤差が小さいことを表し、lower countsはfirst / second / tiesの順で示す。

case_median_across_seeds行は各caseでseed中央値を作ってから、共通case IDでpairにした。

| Pair | Scope | Metric | Count | Mean difference | Median difference | P90 difference | Lower counts | Sources |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
{paired_rows}

## Representative validation cases

MEASURED: difficulty scoreは、各validation caseについて全locked model-seed runのpressure RMSE中央値とした。

median_difficultyはdifficulty scoreが全validation casesの中央値に最も近いcase、high_errorはdifficulty score最大のcaseとし、同値ならcase IDの辞書順で決めた。

このruleはtest caseを再推論せず、同じcaseを全modelと全seedに使う。

| Role | Case ID | Difficulty score | Population median | Sources |
| --- | --- | ---: | ---: | --- |
{representative_rows}

## Pressure fields

{chr(10).join(case_figure_lines)}
## Per-case error distributions

MEASURED: violinは共通test caseごとにseed中央値を取った誤差分布、黒点は各seedのcase-macro平均を示す。 Sources: {sources_text(all_keys, sources)}; `{evidence_artifact}`; `{figures['error_distributions']}`.

![Per-case error distributions](../figures/a1_evidence/{Path(figures['error_distributions']).name})

## Inference profile

MEASURED: profileはrepresentative validation casesだけを使った。

preprocessingはVTU読込、feature/gridまたはgraph構築、checkpoint normalizer適用を含む。

model_forwardはforward callだけ、postprocessing_queryはFNOのgrid-to-point queryまたはpoint output、denormalization、case metric計算を含む。

end_to_endはpreprocessing、forward、postprocessing/queryを直接まとめて測り、checkpoint読込とmodel初期化を含まない。

各componentはunmeasured warmup後に測定したため、filesystem cacheはwarm stateである。

| Model | Seed | Component | Samples | Mean ms | Median ms | P90 ms | Sources |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
{profile_table_rows}

## Accuracy versus latency

MEASURED: y軸は既存frozen test error、x軸はrepresentative validation casesのend-to-end median latencyであり、dataset scopeが異なる値を同じrun単位で並べた。 Sources: {sources_text(all_keys, sources)}; `{evidence_artifact}`; `{profile_artifact}`; `{figures['accuracy_latency']}`.

![Accuracy versus latency](../figures/a1_evidence/{Path(figures['accuracy_latency']).name})

## Interpretation

INTERPRETATION: MLPとGNNのtest error分布は重なっており、このA1だけで一方のarchitectureが一貫して優れるとは判断しない。 Sources: {sources_text(all_keys, sources)}; `{evidence_artifact}`.

INTERPRETATION: FNOはこの固定protocolでMLPとGNNより高いtest errorを示した。 Sources: {sources_text(all_keys, sources)}; `{evidence_artifact}`.

INTERPRETATION: latencyはML inference pipelineの測定であり、CFD solverの実行時間を測っていないためCFD speedupは算出していない。 Sources: {sources_text(all_keys, sources)}; `{profile_artifact}`.

## Hypotheses and limitations

HYPOTHESIS: FNOの誤差にはgrid resolution、invalid-grid処理、grid-to-point query、model capacityの組合せが関係しうるが、このA1は要因を分離していない。 Sources: {sources_text(all_keys, sources)}; `{evidence_artifact}`.

HYPOTHESIS: GNNのlatencyにはgraph preprocessingとmessage passingの両方が関係しうるが、このprofileはmessage-passing block別の内訳を測っていない。 Sources: {sources_text(all_keys, sources)}; `{profile_artifact}`.

この分析はA1で停止する。

A3と記事草稿は作成していない。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--lock", type=Path, default=Path("reports/summaries/a1_checkpoint_lock.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/airfrans_hf_selected/data/Dataset"))
    parser.add_argument("--evidence-json", type=Path, default=Path("reports/summaries/a1_evidence_data.json"))
    parser.add_argument("--profile-json", type=Path, default=Path("reports/summaries/a1_profile.json"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/summaries/a1_representative_predictions.npz"),
    )
    parser.add_argument("--report", type=Path, default=Path("reports/summaries/a1_evidence.md"))
    parser.add_argument("--figure-directory", type=Path, default=Path("reports/figures/a1_evidence"))
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    lock_path = (repository_root / args.lock).resolve()
    dataset_root = (repository_root / args.dataset_root).resolve()
    evidence_json_path = (repository_root / args.evidence_json).resolve()
    profile_json_path = (repository_root / args.profile_json).resolve()
    predictions_path = (repository_root / args.predictions).resolve()
    report_path = (repository_root / args.report).resolve()
    figure_directory = (repository_root / args.figure_directory).resolve()

    sources = load_sources(repository_root, lock_path)
    aggregates = build_aggregate_metrics(sources)
    paired = build_paired_comparisons(sources)
    representative_cases = representative_case_evidence(sources)
    predictions, profile = build_predictions_and_profile(
        repository_root,
        dataset_root,
        sources,
        representative_cases,
    )
    save_prediction_arrays(predictions_path, predictions)
    write_json(profile_json_path, profile)

    relative_prediction_path = predictions_path.relative_to(repository_root).as_posix()
    relative_profile_path = profile_json_path.relative_to(repository_root).as_posix()
    relative_evidence_path = evidence_json_path.relative_to(repository_root).as_posix()
    figures: dict[str, Any] = {"cases": {}}
    for role, evidence in predictions.items():
        figures["cases"][role] = {}
        for seed in SEEDS:
            figure_path = figure_directory / f"a1_{role}_seed_{seed}.png"
            save_case_figure(
                figure_path,
                role,
                seed,
                evidence,
                sources,
                relative_prediction_path,
            )
            figures["cases"][role][str(seed)] = figure_path.relative_to(repository_root).as_posix()
    error_distribution_path = figure_directory / "a1_per_case_error_distributions.png"
    save_error_distribution_figure(error_distribution_path, sources)
    figures["error_distributions"] = error_distribution_path.relative_to(repository_root).as_posix()
    accuracy_latency = build_accuracy_latency(sources, profile)
    accuracy_latency_path = figure_directory / "a1_accuracy_vs_latency.png"
    save_accuracy_latency_figure(
        accuracy_latency_path,
        accuracy_latency,
    )
    figures["accuracy_latency"] = accuracy_latency_path.relative_to(repository_root).as_posix()
    profile_summaries = profile_run_summaries(profile)

    serializable_sources = {
        key: {
            field: value
            for field, value in source.items()
            if field not in {"manifest", "metrics", "timing", "validation_records", "test_records"}
        }
        for key, source in sources.items()
    }
    evidence = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETED_A1_ANALYSIS",
        "retrained_or_tuned": False,
        "raw_test_cases_reopened": False,
        "a3_started": False,
        "article_drafting_started": False,
        "source_registry": serializable_sources,
        "aggregate_test_metrics": aggregates,
        "paired_test_case_comparisons": paired,
        "representative_validation_cases": representative_cases,
        "profile_run_summaries": profile_summaries,
        "accuracy_latency": accuracy_latency,
        "figures": figures,
        "profile_artifact": relative_profile_path,
        "prediction_artifact": relative_prediction_path,
        "lock_path": lock_path.relative_to(repository_root).as_posix(),
        "lock_sha256": file_sha256(lock_path),
    }
    write_json(evidence_json_path, evidence)
    report = build_report(
        sources,
        aggregates,
        paired,
        representative_cases,
        profile_summaries,
        figures,
        relative_evidence_path,
        relative_profile_path,
        relative_prediction_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "report": report_path.relative_to(repository_root).as_posix(),
                "evidence": relative_evidence_path,
                "profile": relative_profile_path,
                "predictions": relative_prediction_path,
                "figures": figures,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
