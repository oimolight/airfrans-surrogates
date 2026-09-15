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

import kaleido
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch import nn

from .data import FinalGridCase, FinalPointCase
from .metrics import bilinear_grid_to_points, case_pressure_metrics
from .model_comparison import (
    MODEL_NAMES,
    _load_cases,
    _normalise_cases,
    locked_source_hashes_match,
    make_model,
)


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
MODEL_COLORS = {
    "mlp": "#0072B2",
    "gnn": "#D55E00",
    "fno": "#009E73",
}
FIGURE_BACKGROUND = "#F4F7FA"
GRID_COLOR = "#D9E1E8"
TEXT_COLOR = "#17212B"


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
    if not locked_source_hashes_match(repository_root, lock["evaluation_source_sha256"]):
        raise RuntimeError("Locked evaluator source changed before evidence analysis")
    sources: dict[str, dict[str, Any]] = {}
    for entry in lock["entries"]:
        run_directory = repository_root / entry["run_directory"]
        manifest = read_json(run_directory / "manifest.json")
        metrics = read_json(run_directory / "metrics.json")
        timing = read_json(run_directory / "timing.json")
        per_case = read_json(run_directory / "per_case_metrics.json")
        checkpoint_path = repository_root / entry["checkpoint_path"]
        if manifest["status"] != "COMPLETED" or manifest["test_evaluation_attempts"] != 1:
            raise RuntimeError(f"Run is not a completed one-time evaluation: {entry['run_id']}")
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
        raise RuntimeError("Checkpoint lock does not contain the complete model-seed matrix")
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
    model = make_model(str(source["model"]), config)
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


def write_figure(figure: go.Figure, path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_image(path, width=width, height=height, scale=2)


def style_figure(figure: go.Figure) -> None:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=FIGURE_BACKGROUND,
        plot_bgcolor="white",
        font={"family": "Avenir, Helvetica Neue, sans-serif", "color": TEXT_COLOR, "size": 13},
        hovermode=False,
    )
    figure.update_xaxes(
        gridcolor=GRID_COLOR,
        linecolor="#9CAAB7",
        showline=True,
        zeroline=False,
    )
    figure.update_yaxes(
        gridcolor=GRID_COLOR,
        linecolor="#9CAAB7",
        showline=True,
        zeroline=False,
    )


def save_case_figure(
    path: Path,
    role: str,
    seed: int,
    evidence: Mapping[str, Any],
) -> None:
    keys = [source_key(model_name, seed) for model_name in MODEL_NAMES]
    positions = np.asarray(evidence["positions"])
    surface_mask = np.asarray(evidence["surface_mask"], dtype=bool)
    reference = np.asarray(evidence["reference"]).reshape(-1)
    inlet_velocity = np.asarray(evidence["inlet_velocity"])
    scale = 0.5 * float(np.dot(inlet_velocity, inlet_velocity))
    reference_scaled = reference / scale
    predictions_scaled = {
        key: np.asarray(evidence["runs"][key]).reshape(-1) / scale
        for key in keys
    }
    pressure_values = [reference_scaled, *predictions_scaled.values()]
    pressure_minimum = min(float(np.min(values)) for values in pressure_values)
    pressure_maximum = max(float(np.max(values)) for values in pressure_values)
    errors = {key: np.abs(value - reference_scaled) for key, value in predictions_scaled.items()}
    error_maximum = float(np.quantile(np.concatenate(list(errors.values())), 0.99))
    subplot_titles = [
        "Reference pressure",
        "MLP prediction",
        "GNN prediction",
        "FNO prediction",
        "Comparison context",
        "MLP absolute error",
        "GNN absolute error",
        "FNO absolute error",
    ]
    figure = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.025,
        vertical_spacing=0.12,
    )
    field_values = [("Reference", reference_scaled)] + [
        (model_name.upper(), predictions_scaled[source_key(model_name, seed)])
        for model_name in MODEL_NAMES
    ]
    for column, (_, values) in enumerate(field_values, start=1):
        figure.add_trace(
            go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode="markers",
                marker={"color": values, "coloraxis": "coloraxis", "size": 4},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=positions[surface_mask, 0],
                y=positions[surface_mask, 1],
                mode="markers",
                marker={"color": "rgba(18, 27, 34, 0.72)", "size": 1.8},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
    figure.update_xaxes(visible=False, row=2, col=1)
    figure.update_yaxes(visible=False, row=2, col=1)
    figure.add_annotation(
        x=0.035,
        y=0.19,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Selection</b> · {role.replace('_', ' ')}<br>"
            f"<b>Split</b> · validation<br>"
            f"<b>Seed</b> · {seed}<br>"
            f"<b>Surface samples</b> · {int(surface_mask.sum())}/{len(surface_mask)}<br>"
            f"<b>Error scale</b> · pooled p99 = {error_maximum:.3g}"
        ),
        showarrow=False,
        align="left",
        font={"size": 13, "color": TEXT_COLOR},
    )
    for column, model_name in enumerate(MODEL_NAMES, start=2):
        key = source_key(model_name, seed)
        figure.add_trace(
            go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode="markers",
                marker={"color": errors[key], "coloraxis": "coloraxis2", "size": 4},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=positions[surface_mask, 0],
                y=positions[surface_mask, 1],
                mode="markers",
                marker={"color": "rgba(18, 27, 34, 0.72)", "size": 1.8},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=column,
        )
    x_minimum, x_maximum = np.min(positions[:, 0]), np.max(positions[:, 0])
    y_minimum, y_maximum = np.min(positions[:, 1]), np.max(positions[:, 1])
    x_padding = max(float(x_maximum - x_minimum) * 0.03, 1e-6)
    y_padding = max(float(y_maximum - y_minimum) * 0.03, 1e-6)
    for panel_index, (row, column) in enumerate(
        [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4)],
        start=1,
    ):
        axis_number = panel_index if panel_index <= 4 else panel_index + 1
        x_reference = "x" if axis_number == 1 else f"x{axis_number}"
        figure.update_xaxes(
            range=[x_minimum - x_padding, x_maximum + x_padding],
            showgrid=False,
            showticklabels=False,
            title_text="x [m]" if row == 2 else None,
            row=row,
            col=column,
        )
        figure.update_yaxes(
            range=[y_minimum - y_padding, y_maximum + y_padding],
            showgrid=False,
            showticklabels=False,
            title_text="y [m]" if column in {1, 2} else None,
            scaleanchor=x_reference,
            scaleratio=1,
            row=row,
            col=column,
        )
    role_label = role.replace("_", "-").title()
    style_figure(figure)
    figure.update_layout(
        title={
            "text": (
                f"<b>{role_label} validation case · seed {seed}</b>"
                f"<br><span style='font-size:12px'>{evidence['case_id']}</span>"
            ),
            "x": 0.025,
            "xanchor": "left",
            "y": 0.985,
            "yanchor": "top",
        },
        margin={"l": 45, "r": 105, "t": 105, "b": 45},
        coloraxis={
            "colorscale": "RdBu",
            "reversescale": True,
            "cmin": pressure_minimum,
            "cmax": pressure_maximum,
            "cmid": 0.0,
            "colorbar": {
                "title": {"text": "(p/ρ)/(q/ρ)", "side": "right"},
                "x": 1.01,
                "y": 0.76,
                "len": 0.38,
                "thickness": 13,
                "outlinewidth": 0,
            },
        },
        coloraxis2={
            "colorscale": "Magma",
            "cmin": 0.0,
            "cmax": error_maximum,
            "colorbar": {
                "title": {"text": "|Δ(p/ρ)|/(q/ρ)<br>p99 cap", "side": "right"},
                "x": 1.01,
                "y": 0.23,
                "len": 0.38,
                "thickness": 13,
                "outlinewidth": 0,
            },
        },
    )
    write_figure(figure, path, width=1600, height=860)


def save_error_distribution_figure(
    path: Path,
    sources: Mapping[str, Mapping[str, Any]],
) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Pressure RMSE", "Surface MAE"),
        horizontal_spacing=0.11,
    )
    labels = [model_name.upper() for model_name in MODEL_NAMES]
    seed_offsets = np.linspace(-0.11, 0.11, len(SEEDS))
    for column, metric in enumerate(METRICS, start=1):
        for model_index, model_name in enumerate(MODEL_NAMES, start=1):
            case_values: dict[str, list[float]] = {}
            for seed in SEEDS:
                key = source_key(model_name, seed)
                for record in sources[key]["test_records"]:
                    case_values.setdefault(str(record["case_id"]), []).append(float(record[metric]))
            distribution = np.asarray([statistics.median(values) for values in case_values.values()])
            figure.add_trace(
                go.Violin(
                    x=np.full(len(distribution), model_index),
                    y=distribution,
                    name=model_name.upper(),
                    legendgroup=model_name,
                    scalegroup=model_name,
                    line={"color": MODEL_COLORS[model_name], "width": 2},
                    fillcolor=MODEL_COLORS[model_name],
                    opacity=0.62,
                    box={"visible": True, "width": 0.18},
                    meanline={"visible": False},
                    points=False,
                    spanmode="hard",
                    width=0.72,
                    showlegend=column == 1,
                    hoverinfo="skip",
                ),
                row=1,
                col=column,
            )
            seed_means = [
                statistics.mean(float(record[metric]) for record in sources[source_key(model_name, seed)]["test_records"])
                for seed in SEEDS
            ]
            figure.add_trace(
                go.Scatter(
                    x=model_index + seed_offsets,
                    y=seed_means,
                    mode="markers+text",
                    text=[str(seed) for seed in SEEDS],
                    textposition="top center",
                    textfont={"size": 10, "color": TEXT_COLOR},
                    marker={
                        "symbol": "diamond",
                        "size": 10,
                        "color": "white",
                        "line": {"color": MODEL_COLORS[model_name], "width": 2.2},
                    },
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=column,
            )
    style_figure(figure)
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, len(labels) + 1)),
        ticktext=labels,
        range=[0.45, len(labels) + 0.55],
        title_text="Model",
    )
    figure.update_yaxes(
        type="log",
        tickmode="array",
        tickvals=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        ticktext=["0.05", "0.1", "0.2", "0.5", "1", "2", "5"],
        title_text="Nondimensional case error · log scale",
    )
    figure.update_layout(
        title={
            "text": (
                "<b>Frozen test error by case</b>"
                "<br><span style='font-size:12px'>Violin and box: 50 case medians across seeds · diamonds: seed-level case means</span>"
            ),
            "x": 0.025,
            "xanchor": "left",
            "y": 0.98,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.16,
            "xanchor": "right",
            "x": 1.0,
            "title": {"text": "Model"},
        },
        margin={"l": 80, "r": 35, "t": 135, "b": 65},
    )
    write_figure(figure, path, width=1320, height=650)


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
    metrics = (
        "test_pressure_rmse_nondimensional_macro_mean",
        "test_surface_mae_nondimensional_macro_mean",
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Pressure RMSE", "Surface MAE"),
        horizontal_spacing=0.11,
    )
    seed_symbols = {17: "circle", 29: "diamond", 41: "square"}
    for column, metric in enumerate(metrics, start=1):
        metric_values = [float(row[metric]) for row in rows]
        y_padding = max((max(metric_values) - min(metric_values)) * 0.18, 0.02)
        for model_name in MODEL_NAMES:
            model_rows = sorted(
                (row for row in rows if row["model"] == model_name),
                key=lambda row: int(row["seed"]),
            )
            x_values = [float(row["representative_case_end_to_end_latency_median_ms"]) for row in model_rows]
            y_values = [float(row[metric]) for row in model_rows]
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="markers",
                    name=model_name.upper(),
                    legendgroup=model_name,
                    marker={
                        "size": 12,
                        "color": MODEL_COLORS[model_name],
                        "symbol": [seed_symbols[int(row["seed"])] for row in model_rows],
                        "line": {"color": "white", "width": 1.5},
                    },
                    showlegend=column == 1,
                    cliponaxis=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=[statistics.median(x_values)],
                    y=[statistics.median(y_values)],
                    mode="markers",
                    marker={
                        "symbol": "circle-open",
                        "size": 22,
                        "color": MODEL_COLORS[model_name],
                        "line": {"color": MODEL_COLORS[model_name], "width": 2.5},
                    },
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=column,
            )
        figure.update_yaxes(
            range=[min(metric_values) - y_padding, max(metric_values) + y_padding],
            row=1,
            col=column,
        )
        axis_suffix = "" if column == 1 else str(column)
        figure.add_annotation(
            x=0.02,
            y=0.04,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text="↙ lower error and latency",
            showarrow=False,
            font={"size": 11, "color": "#5B6873"},
            align="left",
        )
    style_figure(figure)
    figure.update_xaxes(
        type="log",
        tickmode="array",
        tickvals=[200, 300, 500, 1000, 1500],
        ticktext=["200", "300", "500", "1,000", "1,500"],
        title_text="Representative validation-case latency [ms] · log scale",
    )
    figure.update_yaxes(title_text="Frozen test error · lower is better")
    figure.update_layout(
        title={
            "text": (
                "<b>Accuracy versus end-to-end inference latency</b>"
                "<br><span style='font-size:12px'>Seed symbols: ● 17 · ◆ 29 · ■ 41 · large rings: model medians</span>"
            ),
            "x": 0.025,
            "xanchor": "left",
            "y": 0.98,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.16,
            "xanchor": "right",
            "x": 1.0,
            "title": {"text": "Model"},
        },
        margin={"l": 85, "r": 35, "t": 135, "b": 75},
    )
    write_figure(figure, path, width=1320, height=650)


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
                    f"MEASURED: Reference pressure, prediction, and absolute error are shown for the same validation case. "
                    f"Sources: {sources_text(keys, sources)}; `{prediction_artifact}`; `{path}`.",
                    "",
                    f"![{role} seed {seed}](../figures/model_comparison/{Path(path).name})",
                    "",
                ]
            )
    return f"""# AirfRANS model-comparison evidence

Status: COMPLETED MODEL-COMPARISON ANALYSIS; NO RETRAINING OR TUNING

## Evidence labels

**MEASURED** denotes values computed directly from completed run artifacts, validation inference using locked checkpoints, or profile samples saved by this analysis.

**INTERPRETATION** denotes explanations limited to what can be inferred from measured values.

**HYPOTHESIS** denotes candidate explanations that cannot be established without additional experiments.

The Sources column in every numeric table resolves to run IDs and artifact paths in the following source registry.

## Source registry

| Key | run_id | Artifact paths |
| --- | --- | --- |
{source_rows}

The analysis artifact is `{evidence_artifact}`, the profile artifact is `{profile_artifact}`, and the field artifact is `{prediction_artifact}`.

## Aggregate test metrics

MEASURED: Seeds in the frozen matrix are shown in individual rows. The case_median_across_seeds rows were aggregated after taking the median across seeds for each shared case.

P90 uses NumPy's linear quantile method, and lower error is better.

| Model | Scope | Metric | Count | Mean | Median | P90 | Sources |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
{aggregate_rows}

## Paired per-case comparisons

MEASURED: Differences are the first model minus the second model.

A negative difference means that the first model has lower error. Lower counts are ordered as first / second / ties.

The case_median_across_seeds rows were paired by shared case ID after taking the median across seeds for each case.

| Pair | Scope | Metric | Count | Mean difference | Median difference | P90 difference | Lower counts | Sources |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
{paired_rows}

## Representative validation cases

MEASURED: The difficulty score for each validation case is the median pressure RMSE across all locked model-seed runs.

The median_difficulty case has the difficulty score closest to the median across all validation cases. The high_error case has the largest difficulty score. Ties are resolved by lexicographic case ID order.

This rule does not rerun inference on test cases and uses the same case for every model and seed.

| Role | Case ID | Difficulty score | Population median | Sources |
| --- | --- | ---: | ---: | --- |
{representative_rows}

## Pressure fields

MEASURED: Every pressure panel within a figure shares one color scale. Absolute-error panels share a color scale capped at the pooled pointwise p99 for visibility; larger displayed values use the top color, while all reported metrics remain uncapped.

{chr(10).join(case_figure_lines)}
## Per-case error distributions

MEASURED: Violin shapes and internal boxes show the distribution of 50 shared test-case errors after taking the median across seeds for each case. Seed-labeled diamonds show the case-macro mean for each seed. The vertical axis is logarithmic. Sources: {sources_text(all_keys, sources)}; `{evidence_artifact}`; `{figures['error_distributions']}`.

![Per-case error distributions](../figures/model_comparison/{Path(figures['error_distributions']).name})

## Inference profile

MEASURED: Profiling used only the representative validation cases.

Preprocessing includes VTU loading, feature and grid or graph construction, and application of the checkpoint normalizer.

model_forward includes only the forward call. postprocessing_query includes the FNO grid-to-point query or point output, denormalization, and case-metric calculation.

end_to_end directly measures preprocessing, forward, and postprocessing/query together; it excludes checkpoint loading and model initialization.

Each component was measured after an unmeasured warmup, so the filesystem cache was warm.

| Model | Seed | Component | Samples | Mean ms | Median ms | P90 ms | Sources |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
{profile_table_rows}

## Accuracy versus latency

MEASURED: The y-axis is the existing frozen test error, and the x-axis is median end-to-end latency on representative validation cases. Point shapes identify seeds, and large open rings show model medians. Values with different dataset scopes are shown together at the run level. Sources: {sources_text(all_keys, sources)}; `{evidence_artifact}`; `{profile_artifact}`; `{figures['accuracy_latency']}`.

![Accuracy versus latency](../figures/model_comparison/{Path(figures['accuracy_latency']).name})

## Interpretation

INTERPRETATION: The MLP and GNN test-error distributions overlap, so this model comparison alone does not show that either architecture is consistently better. Sources: {sources_text(all_keys, sources)}; `{evidence_artifact}`.

INTERPRETATION: Under this fixed protocol, the FNO had higher test error than the MLP and GNN. Sources: {sources_text(all_keys, sources)}; `{evidence_artifact}`.

INTERPRETATION: Latency measures the ML inference pipeline. CFD speedup was not calculated because CFD solver runtime was not measured. Sources: {sources_text(all_keys, sources)}; `{profile_artifact}`.

## Hypotheses and limitations

HYPOTHESIS: FNO error may reflect a combination of grid resolution, invalid-grid handling, the grid-to-point query, and model capacity, but this model comparison does not isolate these factors. Sources: {sources_text(all_keys, sources)}; `{evidence_artifact}`.

HYPOTHESIS: GNN latency may reflect both graph preprocessing and message passing, but this profile does not measure separate message-passing blocks. Sources: {sources_text(all_keys, sources)}; `{profile_artifact}`.

This analysis stops at model comparison.

This analysis command does not create an edge-feature ablation.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--lock", type=Path, default=Path("reports/summaries/model_comparison_checkpoint_lock.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/airfrans_hf_selected/data/Dataset"))
    parser.add_argument("--evidence-json", type=Path, default=Path("reports/summaries/model_comparison_evidence_data.json"))
    parser.add_argument("--profile-json", type=Path, default=Path("reports/summaries/model_comparison_profile.json"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/summaries/model_comparison_representative_predictions.npz"),
    )
    parser.add_argument("--report", type=Path, default=Path("reports/summaries/model_comparison_evidence.md"))
    parser.add_argument("--figure-directory", type=Path, default=Path("reports/figures/model_comparison"))
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
    accuracy_latency = build_accuracy_latency(sources, profile)
    kaleido.start_sync_server()
    try:
        for role, evidence in predictions.items():
            figures["cases"][role] = {}
            for seed in SEEDS:
                figure_path = figure_directory / f"{role}_seed_{seed}.png"
                save_case_figure(
                    figure_path,
                    role,
                    seed,
                    evidence,
                )
                figures["cases"][role][str(seed)] = figure_path.relative_to(repository_root).as_posix()
        error_distribution_path = figure_directory / "per_case_error_distributions.png"
        save_error_distribution_figure(error_distribution_path, sources)
        figures["error_distributions"] = error_distribution_path.relative_to(repository_root).as_posix()
        accuracy_latency_path = figure_directory / "accuracy_vs_latency.png"
        save_accuracy_latency_figure(
            accuracy_latency_path,
            accuracy_latency,
        )
        figures["accuracy_latency"] = accuracy_latency_path.relative_to(repository_root).as_posix()
    finally:
        kaleido.stop_sync_server()
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
        "status": "COMPLETED_MODEL_COMPARISON_ANALYSIS",
        "retrained_or_tuned": False,
        "raw_test_cases_reopened": False,
        "edge_feature_ablation_started": False,
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
