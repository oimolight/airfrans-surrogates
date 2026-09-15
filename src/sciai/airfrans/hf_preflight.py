from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi


REPO_ID = "OneScience-Group/airfrans"
REPO_TYPE = "dataset"
SOURCE_MANIFEST_PATH = "data/Dataset/manifest.json"
EXPECTED_FROZEN_MANIFEST_SHA256 = "8c532a15c2af6538177ae9885de8978553b162c4eb788d33a2998c9edc447014"
EXPECTED_CASE_SUFFIXES = ("_aerofoil.vtp", "_freestream.vtp", "_internal.vtu")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_case_ids(path: Path) -> tuple[dict[str, object], list[str]]:
    actual_sha256 = file_sha256(path)
    if actual_sha256 != EXPECTED_FROZEN_MANIFEST_SHA256:
        raise RuntimeError(
            f"Frozen selection hash changed: {actual_sha256}; expected {EXPECTED_FROZEN_MANIFEST_SHA256}"
        )
    selected = json.loads(path.read_text(encoding="utf-8"))
    splits = selected["splits"]
    case_ids = [
        *splits["train"],
        *splits["validation"],
        *splits["id_test"],
        *splits["geometry_ood_test"],
    ]
    if len(case_ids) != 250 or len(set(case_ids)) != 250:
        raise RuntimeError("Frozen selection must contain exactly 250 unique case IDs")
    if selected["protocol"]["geometry_ood_status"] != "NOT_DEFINED":
        raise RuntimeError("Frozen geometry-OOD status is not NOT_DEFINED")
    return selected, sorted(case_ids)


def safe_relative_path(remote_path: str) -> PurePosixPath:
    raw_parts = remote_path.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise RuntimeError(f"Unsafe remote path: {remote_path}")
    path = PurePosixPath(remote_path)
    if path.is_absolute():
        raise RuntimeError(f"Unsafe remote path: {remote_path}")
    if len(path.parts) < 3 or path.parts[:2] != ("data", "Dataset"):
        raise RuntimeError(f"Remote path is outside data/Dataset: {remote_path}")
    return path


def safe_destination_path(destination: Path, remote_path: PurePosixPath) -> Path:
    remote_path = safe_relative_path(str(remote_path))
    destination_root = destination.resolve()
    local_path = destination.joinpath(*remote_path.parts).resolve()
    if os.path.commonpath((destination_root, local_path)) != str(destination_root):
        raise RuntimeError(f"Resolved path escapes destination: {remote_path}")
    return local_path


def resolve_remote_files(
    selected_case_ids: list[str],
    siblings: list[object],
    destination: Path,
) -> tuple[list[dict[str, object]], int]:
    by_path = {sibling.rfilename: sibling for sibling in siblings}
    selected_files = []
    selected_paths = {SOURCE_MANIFEST_PATH}

    for case_id in selected_case_ids:
        for suffix in EXPECTED_CASE_SUFFIXES:
            selected_paths.add(f"data/Dataset/{case_id}/{case_id}{suffix}")

    for remote_path in sorted(selected_paths):
        path = safe_relative_path(remote_path)
        safe_destination_path(destination, path)
        sibling = by_path.get(remote_path)
        if sibling is None:
            raise RuntimeError(f"Required remote file is missing: {remote_path}")
        size = getattr(sibling, "size", None)
        if not isinstance(size, int) or size < 0:
            raise RuntimeError(f"Remote file has no valid size: {remote_path}")
        lfs = getattr(sibling, "lfs", None)
        selected_files.append(
            {
                "remote_path": remote_path,
                "size_bytes": size,
                "blob_id": getattr(sibling, "blob_id", None),
                "lfs_sha256": getattr(lfs, "sha256", None) if lfs is not None else None,
                "destination_path": str(destination.joinpath(*path.parts)),
            }
        )

    remote_case_file_count = sum(
        1
        for remote_path in by_path
        if len(PurePosixPath(remote_path).parts) == 4
        and PurePosixPath(remote_path).parts[:2] == ("data", "Dataset")
        and PurePosixPath(remote_path).suffix in {".vtp", ".vtu"}
    )
    return selected_files, remote_case_file_count - len(selected_case_ids) * len(EXPECTED_CASE_SUFFIXES)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(path: Path, plan: dict[str, object]) -> None:
    counts = plan["counts"]
    storage = plan["storage"]
    report = f"""# AirfRANS selective download preflight

Status: AWAITING APPROVAL

CFD case files downloaded by this step: no

## Source and destination

- Hugging Face dataset: `{plan["repo_id"]}`
- Pinned revision: `{plan["revision"]}`
- Frozen selection: `{plan["frozen_selection_path"]}`
- Frozen selection SHA-256: `{plan["frozen_selection_sha256"]}`
- Destination: `{plan["destination"]}`

## Planned files

- Selected unique cases: {counts["selected_unique_cases"]}
- Files per case: 3 (`_aerofoil.vtp`, `_freestream.vtp`, `_internal.vtu`)
- Selected case files: {counts["case_files"]}
- Source manifest files: {counts["source_manifest_files"]}
- Total files: {counts["total_files"]}
- Unselected case files excluded: {counts["unselected_case_files_excluded"]}
- Missing selected cases: {counts["missing_selected_cases"]}
- Unknown remote sizes: {counts["unknown_sizes"]}

## Storage

- Expected case bytes: {storage["case_bytes"]}
- Expected source manifest bytes: {storage["source_manifest_bytes"]}
- Expected total bytes: {storage["total_bytes"]}
- Available bytes before download: {storage["available_bytes"]}
- Expected available bytes after download: {storage["expected_available_after_download_bytes"]}

## Path safety

Every planned remote path is relative, contains neither `.` nor `..`, begins with `data/Dataset/`, and resolves inside the destination. Case files are selected only when the immediate parent directory exactly matches a frozen case ID. No substring or wildcard case matching is used.

The preserved ISIR partial is outside the destination and is not deleted or read by the planned download.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("data/manifests/airfrans_selected_cases.json"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/airfrans_hf_selected"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/airfrans_hf_download_plan.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/airfrans_download_preflight.md"),
    )
    args = parser.parse_args()

    _, selected_case_ids = load_frozen_case_ids(args.selection)
    info = HfApi().dataset_info(REPO_ID, files_metadata=True)
    if not info.sha:
        raise RuntimeError("Hugging Face metadata did not expose a revision SHA")
    files, unselected_count = resolve_remote_files(selected_case_ids, list(info.siblings), args.destination)
    source_manifest = next(item for item in files if item["remote_path"] == SOURCE_MANIFEST_PATH)
    case_files = [item for item in files if item["remote_path"] != SOURCE_MANIFEST_PATH]
    available_bytes = shutil.disk_usage(Path.cwd()).free
    total_bytes = sum(int(item["size_bytes"]) for item in files)
    plan = {
        "status": "AWAITING_APPROVAL",
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "revision": info.sha,
        "frozen_selection_path": str(args.selection),
        "frozen_selection_sha256": file_sha256(args.selection),
        "destination": str(args.destination),
        "selection_rule": "Exact remote paths derived only from the 250 frozen unique case IDs; no wildcards.",
        "counts": {
            "selected_unique_cases": len(selected_case_ids),
            "case_files": len(case_files),
            "source_manifest_files": 1,
            "total_files": len(files),
            "unselected_case_files_excluded": unselected_count,
            "missing_selected_cases": 0,
            "unknown_sizes": 0,
        },
        "storage": {
            "case_bytes": sum(int(item["size_bytes"]) for item in case_files),
            "source_manifest_bytes": int(source_manifest["size_bytes"]),
            "total_bytes": total_bytes,
            "available_bytes": available_bytes,
            "expected_available_after_download_bytes": available_bytes - total_bytes,
        },
        "path_safety": {
            "all_paths_relative": True,
            "dot_segments_rejected": True,
            "all_paths_under_data_dataset": True,
            "all_destinations_within_root": True,
            "case_parent_exact_match_required": True,
        },
        "files": files,
    }
    write_json(args.output, plan)
    write_report(args.report, plan)
    print(
        json.dumps(
            {
                "revision": info.sha,
                "counts": plan["counts"],
                "storage": plan["storage"],
                "destination": plan["destination"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
