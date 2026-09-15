from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from huggingface_hub import snapshot_download

from .hf_preflight import (
    EXPECTED_CASE_SUFFIXES,
    EXPECTED_FROZEN_MANIFEST_SHA256,
    REPO_ID,
    REPO_TYPE,
    SOURCE_MANIFEST_PATH,
    file_sha256,
    load_frozen_case_ids,
    safe_destination_path,
    safe_relative_path,
)


EXPECTED_PLAN_SHA256 = "3b10ef376da0f735b8bab43ad433939ce205849a52d1cd44316c50299a0bf7f8"
EXPECTED_REVISION = "f99b37789d0c76e02b923b14b23dcd6389f0d895"
EXPECTED_DESTINATION = Path("data/raw/airfrans_hf_selected")


def load_approved_plan(plan_path: Path, selection_path: Path) -> dict[str, object]:
    plan_sha256 = file_sha256(plan_path)
    if plan_sha256 != EXPECTED_PLAN_SHA256:
        raise RuntimeError(f"Download plan hash changed: {plan_sha256}; expected {EXPECTED_PLAN_SHA256}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _, case_ids = load_frozen_case_ids(selection_path)
    expected_paths = {SOURCE_MANIFEST_PATH}
    for case_id in case_ids:
        expected_paths.update(
            f"data/Dataset/{case_id}/{case_id}{suffix}" for suffix in EXPECTED_CASE_SUFFIXES
        )

    files = plan.get("files")
    if not isinstance(files, list):
        raise RuntimeError("Download plan files must be a list")
    remote_paths = [item.get("remote_path") for item in files if isinstance(item, dict)]
    if len(files) != 751 or len(remote_paths) != 751 or set(remote_paths) != expected_paths:
        raise RuntimeError("Download plan does not match the 751 exact approved paths")
    if len(set(remote_paths)) != len(remote_paths):
        raise RuntimeError("Download plan contains duplicate remote paths")

    if plan.get("repo_id") != REPO_ID or plan.get("repo_type") != REPO_TYPE:
        raise RuntimeError("Download plan repository changed")
    if plan.get("revision") != EXPECTED_REVISION:
        raise RuntimeError("Download plan revision changed")
    if plan.get("frozen_selection_sha256") != EXPECTED_FROZEN_MANIFEST_SHA256:
        raise RuntimeError("Download plan selection hash changed")
    if Path(str(plan.get("destination"))) != EXPECTED_DESTINATION:
        raise RuntimeError("Download plan destination changed")

    total_bytes = 0
    for item in files:
        remote_path = str(item["remote_path"])
        safe_destination_path(EXPECTED_DESTINATION, safe_relative_path(remote_path))
        size_bytes = item.get("size_bytes")
        if not isinstance(size_bytes, int) or size_bytes < 0:
            raise RuntimeError(f"Invalid planned size: {remote_path}")
        total_bytes += size_bytes
    if total_bytes != 3_745_583_013:
        raise RuntimeError(f"Download plan byte total changed: {total_bytes}")
    return plan


def verify_download(plan: dict[str, object]) -> tuple[int, int]:
    destination = Path(str(plan["destination"]))
    files = plan["files"]
    planned_paths = {str(item["remote_path"]) for item in files}
    actual_paths = {
        path.relative_to(destination).as_posix()
        for path in (destination / "data" / "Dataset").rglob("*")
        if path.is_file() and (path.suffix in {".vtp", ".vtu"} or path.name == "manifest.json")
    }
    if actual_paths != planned_paths:
        missing = sorted(planned_paths - actual_paths)
        extra = sorted(actual_paths - planned_paths)
        raise RuntimeError(f"Downloaded path set differs: missing={missing[:3]}, extra={extra[:3]}")

    actual_bytes = 0
    for item in files:
        remote_path = safe_relative_path(str(item["remote_path"]))
        local_path = safe_destination_path(destination, remote_path)
        actual_size = local_path.stat().st_size
        if actual_size != item["size_bytes"]:
            raise RuntimeError(
                f"Downloaded size differs for {remote_path}: {actual_size}; expected {item['size_bytes']}"
            )
        actual_bytes += actual_size
    return len(actual_paths), actual_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("data/manifests/airfrans_hf_download_plan.json"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("data/manifests/airfrans_selected_cases.json"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("reports/airfrans_download_result.json"),
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = load_approved_plan(args.plan, args.selection)
    if not args.execute:
        raise RuntimeError("Download not started: pass --execute only after explicit approval")
    if args.max_workers < 1:
        raise RuntimeError("--max-workers must be at least 1")

    remote_paths = [str(item["remote_path"]) for item in plan["files"]]
    snapshot_download(
        repo_id=str(plan["repo_id"]),
        repo_type=str(plan["repo_type"]),
        revision=str(plan["revision"]),
        local_dir=Path(str(plan["destination"])),
        allow_patterns=remote_paths,
        max_workers=args.max_workers,
    )
    file_count, actual_bytes = verify_download(plan)
    result = {
        "status": "DOWNLOADED_AND_VERIFIED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": plan["repo_id"],
        "revision": plan["revision"],
        "plan_path": str(args.plan),
        "plan_sha256": file_sha256(args.plan),
        "frozen_selection_sha256": file_sha256(args.selection),
        "destination": plan["destination"],
        "verified_file_count": file_count,
        "verified_bytes": actual_bytes,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
