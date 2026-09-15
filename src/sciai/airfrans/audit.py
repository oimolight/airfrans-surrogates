from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from airfrans import Simulation


SPLIT_SEED = "airfrans-m1-v1"
SELECTION_SCRIPT_VERSION = "airfrans-selection-v1"


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def case_metadata(case_id: str) -> dict[str, object]:
    parts = case_id.split("_")
    if len(parts) < 7 or parts[0:2] != ["airFoil2D", "SST"]:
        raise ValueError(f"Unsupported official case name: {case_id}")
    return {
        "case_id": case_id,
        "inlet_velocity_from_name": float(parts[2]),
        "angle_of_attack_degrees_from_name": float(parts[3]),
        "geometry_group": "_".join(parts[4:]),
        "metadata_contract": "airfrans.Simulation parses conditions at tokens 2 and 3; remaining tokens identify NACA geometry",
    }


def source_inventory(source_manifest: dict[str, list[str]]) -> dict[str, object]:
    memberships: dict[str, list[str]] = defaultdict(list)
    for source_partition, case_ids in source_manifest.items():
        for case_id in case_ids:
            memberships[case_id].append(source_partition)
    cases = []
    for case_id in sorted(memberships):
        metadata = case_metadata(case_id)
        metadata["source_partitions"] = sorted(memberships[case_id])
        cases.append(metadata)
    return {
        "dataset": "AirfRANS pre-processed Dataset.zip",
        "case_count": len(cases),
        "source_manifest_sha256": canonical_hash(source_manifest),
        "cases": cases,
    }


def _rank(group: str, purpose: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{purpose}:{group}".encode()).hexdigest()


def build_split(source_manifest: dict[str, list[str]], audit_case_id: str) -> dict[str, object]:
    scarce_ids = set(source_manifest["scarce_train"])
    official_test_ids = set(source_manifest["full_test"])
    if audit_case_id not in scarce_ids:
        raise ValueError("Audit case is not in official scarce_train")

    scarce_groups: dict[str, list[str]] = defaultdict(list)
    for case_id in scarce_ids:
        scarce_groups[str(case_metadata(case_id)["geometry_group"])].append(case_id)
    audit_group = str(case_metadata(audit_case_id)["geometry_group"])

    validation_groups: set[str] = set()
    validation_count = 0
    candidates = sorted((group for group in scarce_groups if group != audit_group), key=lambda group: _rank(group, "validation"))
    for group in candidates:
        group_size = len(scarce_groups[group])
        if abs(validation_count + group_size - 40) < abs(validation_count - 40):
            validation_groups.add(group)
            validation_count += group_size

    train_ids = sorted(case_id for group, ids in scarce_groups.items() if group not in validation_groups for case_id in ids)
    validation_ids = sorted(case_id for group in validation_groups for case_id in scarce_groups[group])
    pool_groups = set(scarce_groups)

    eligible_test = [
        case_id
        for case_id in official_test_ids
        if str(case_metadata(case_id)["geometry_group"]) not in pool_groups
    ]
    test_ids = sorted(eligible_test, key=lambda case_id: _rank(case_id, "test"))[:50]
    if len(test_ids) < 50:
        raise RuntimeError(f"Only {len(test_ids)} geometry-disjoint official test cases are available")

    split = {
        "algorithm": "airfrans-group-split-v1",
        "seed": SPLIT_SEED,
        "source_partition_note": "official scarce_train is distinct from local train",
        "audit_case_id": audit_case_id,
        "audit_geometry_group": audit_group,
        "local_train": train_ids,
        "local_validation": validation_ids,
        "local_test": sorted(test_ids),
        "excluded_sealed_official_test": sorted(official_test_ids - set(test_ids)),
        "calibration": {"status": "not_applicable_for_airfrans_spec"},
        "test_sealed": True,
        "target_counts": {"train": 160, "validation": 40, "test": 50},
        "actual_counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
        },
    }
    split["split_sha256"] = canonical_hash(split)
    return split


def build_selected_cases(
    source_manifest: dict[str, list[str]],
    source_file_sha256: str | None = None,
) -> dict[str, object]:
    relevant_ids = sorted(set(source_manifest["scarce_train"]) | set(source_manifest["full_test"]))
    unparsed = []
    for case_id in relevant_ids:
        try:
            case_metadata(case_id)
        except (TypeError, ValueError) as error:
            unparsed.append({"case_id": case_id, "reason": str(error)})
    if unparsed:
        raise ValueError(f"Cannot preserve geometry groups for {len(unparsed)} unparsed cases")

    audit_case_id = sorted(source_manifest["scarce_train"])[0]
    split = build_split(source_manifest, audit_case_id)
    selected = {
        "schema_version": 1,
        "protocol": {
            "algorithm": split["algorithm"],
            "seed": split["seed"],
            "selection_script": "src/sciai/airfrans/selection.py",
            "selection_script_version": SELECTION_SCRIPT_VERSION,
            "git_commit": None,
            "git_commit_status": "not_available_not_git_repository",
            "source_partition_note": split["source_partition_note"],
            "whole_geometry_groups_preserved": True,
            "exact_geometry_key": "All underscore-delimited case ID tokens from index 4 onward.",
            "selection_rule": (
                "Use all 200 official scarce_train cases. Keep the first sorted case's exact geometry group in train. "
                "Rank every other exact geometry group by SHA-256(seed + ':validation:' + geometry_key), then add "
                "whole groups when doing so reduces the distance from the validation target of 40; assign the rest "
                "to train. From official full_test, retain cases whose exact geometry key is absent from the 200-case "
                "pool, rank by SHA-256(seed + ':test:' + case_id), take the first 50, then sort IDs for storage."
            ),
            "id_test_definition": (
                "Fixed 50 cases from the official full/scarce interpolation test partition: distribution-ID with "
                "exact geometries unseen in train and validation."
            ),
            "geometry_ood_status": "NOT_DEFINED",
            "geometry_ood_reason": (
                "The project specification requires a geometry holdout threshold to be fixed after metadata audit, "
                "but no threshold is currently approved. The official full/scarce test partition must not be relabeled "
                "as geometry-OOD."
            ),
        },
        "source": {
            "manifest": "data/airfrans_hf/data/Dataset/manifest.json",
            "manifest_file_sha256": source_file_sha256,
            "manifest_canonical_json_sha256": canonical_hash(source_manifest),
        },
        "splits": {
            "train": split["local_train"],
            "validation": split["local_validation"],
            "id_test": split["local_test"],
            "geometry_ood_test": [],
        },
        "counts": {
            "train": len(split["local_train"]),
            "validation": len(split["local_validation"]),
            "id_test": len(split["local_test"]),
            "geometry_ood_test": "NOT_DEFINED",
            "geometry_ood_case_list_length": 0,
        },
        "total_unique_cases": len(
            set(split["local_train"]) | set(split["local_validation"]) | set(split["local_test"])
        ),
        "unparsed_geometry_metadata": unparsed,
    }
    selected["selection_sha256"] = canonical_hash(selected)
    return selected


def array_schema(array: np.ndarray) -> dict[str, object]:
    numeric = np.issubdtype(array.dtype, np.number)
    finite = np.isfinite(array) if numeric else None
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
        "nan_count": int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0,
        "min": float(array[finite].min()) if numeric and finite is not None and finite.any() else None,
        "max": float(array[finite].max()) if numeric and finite is not None and finite.any() else None,
    }


def mesh_schema(mesh: pv.DataSet) -> dict[str, object]:
    return {
        "point_count": int(mesh.n_points),
        "cell_count": int(mesh.n_cells),
        "points": array_schema(np.asarray(mesh.points)),
        "point_data": {name: array_schema(np.asarray(mesh.point_data[name])) for name in sorted(mesh.point_data)},
        "cell_data": {name: array_schema(np.asarray(mesh.cell_data[name])) for name in sorted(mesh.cell_data)},
    }


def find_dataset_root(extract_root: Path) -> Path:
    matches = list(extract_root.rglob("manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one extracted manifest, found {len(matches)}")
    return matches[0].parent


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-root", type=Path, default=Path("data/raw/airfrans/extracted"))
    parser.add_argument("--fetch-metadata", type=Path, default=Path("data/raw/airfrans/fetch_metadata.json"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()

    dataset_root = find_dataset_root(args.extract_root)
    manifest_path = dataset_root / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_keys = {"scarce_train", "full_test"}
    if not required_keys.issubset(source_manifest):
        raise RuntimeError(f"Missing actual manifest keys: {sorted(required_keys - source_manifest.keys())}")

    audit_case_id = sorted(source_manifest["scarce_train"])[0]
    inventory = source_inventory(source_manifest)
    split = build_split(source_manifest, audit_case_id)
    write_json(args.reports / "data_manifest.json", inventory)
    write_json(args.reports / "split_manifest.json", split)

    case_directory = dataset_root / audit_case_id
    internal_path = case_directory / f"{audit_case_id}_internal.vtu"
    airfoil_path = case_directory / f"{audit_case_id}_aerofoil.vtp"
    internal = pv.read(internal_path)
    airfoil = pv.read(airfoil_path)
    simulation = Simulation(root=str(dataset_root), name=audit_case_id)
    fetch_metadata = json.loads(args.fetch_metadata.read_text(encoding="utf-8"))

    audit = {
        "status": "observed_one_pool_case",
        "source": {
            **fetch_metadata,
            "license": "ODbL-1.0",
            "library_license": "MIT",
            "airfrans_version": version("airfrans"),
            "archive_path": "data/raw/airfrans/Dataset.zip",
            "archive_size_bytes": Path("data/raw/airfrans/Dataset.zip").stat().st_size,
        },
        "case": {
            **case_metadata(audit_case_id),
            "source_partition": "official_scarce_train",
            "local_role": "train",
            "files": {
                internal_path.name: internal_path.stat().st_size,
                airfoil_path.name: airfoil_path.stat().st_size,
            },
        },
        "raw_schema": {
            "internal_vtu": mesh_schema(internal),
            "aerofoil_vtp": mesh_schema(airfoil),
        },
        "confirmed_meanings": {
            "internal.points[:, :2]": "x/y position in metres; official AirfRANS documentation",
            "internal.point_data.U[:, :2]": "solver target velocity in m/s; prohibited as model input",
            "internal.point_data.p": "solver target pressure divided by specific mass in m^2/s^2",
            "internal.point_data.nut": "solver target turbulent kinematic viscosity in m^2/s; out of project output scope",
            "internal.point_data.implicit_distance": "geometry-derived distance field; sign convention observed by official loader but not independently validated",
            "aerofoil.point_data.Normals[:, :2]": "official preprocessing documents inward-pointing surface normals",
            "case_name tokens 2/3": "inlet velocity and angle of attack; official Simulation parser",
        },
        "input_policy": {
            "allowed_after_geometry_validation": ["position", "input_velocity", "implicit_distance", "geometry-derived surface mask", "geometry-derived normals"],
            "target": "pressure p/rho only",
            "prohibited": ["velocity U", "pressure p", "turbulent viscosity nut", "Simulation.surface", "Simulation.normals"],
            "reason": "AirfRANS 0.1.5.1 derives Simulation.surface from target U[:, 0] == 0 and derives Simulation.normals through that mask.",
        },
        "loader_observations": {
            "position": array_schema(simulation.position),
            "input_velocity": array_schema(simulation.input_velocity),
            "sdf": array_schema(simulation.sdf),
            "surface_label_derived_do_not_use": array_schema(simulation.surface),
            "normals_label_derived_do_not_use": array_schema(simulation.normals),
            "pressure_target": array_schema(simulation.pressure),
        },
        "unknown_or_unverified": [
            "independent physical validation of coordinate orientation and origin",
            "independent validation of implicit_distance sign",
            "geometry-only surface-mask implementation",
            "geometry-only transfer of aerofoil normals to volume points",
            "ML library GPU support and performance",
        ],
        "test_data_access": "No official test mesh was extracted or opened; only case IDs and name-derived geometry metadata were used to seal the test split.",
        "prediction_result": "none; M1 contains no model or prediction",
    }
    write_json(args.reports / "data_audit.json", audit)
    print(json.dumps({"audit_case_id": audit_case_id, "split_counts": split["actual_counts"], "split_sha256": split["split_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
