from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch


REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "run_id",
        "experiment_id",
        "model",
        "seed",
        "git_commit",
        "git_dirty",
        "python_version",
        "pytorch_version",
        "device",
        "source_airfrans_manifest_sha256",
        "selected_case_manifest_sha256",
        "config_sha256",
        "actual_case_counts",
        "status",
        "failure_reason",
        "checkpoint_path",
    }
)
RUN_ARTIFACT_FILES = (
    "manifest.json",
    "resolved_config.json",
    "training_history.json",
    "metrics.json",
    "per_case_metrics.json",
    "timing.json",
)
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED"})
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class CompletedRunError(FileExistsError):
    """Raised when code tries to replace a completed run manifest."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def config_sha256(config: object) -> str:
    """Return a deterministic SHA-256 hash of a JSON-compatible config."""
    return hashlib.sha256(_json_bytes(config)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_directory(runs_root: Path, experiment_id: str) -> tuple[str, Path]:
    if not _SAFE_ID.fullmatch(experiment_id):
        raise ValueError("experiment_id must contain only letters, digits, '.', '_', and '-'")
    runs_root.mkdir(parents=True, exist_ok=True)
    while True:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{experiment_id}-{timestamp}-{uuid4().hex[:12]}"
        run_directory = runs_root / run_id
        try:
            run_directory.mkdir()
        except FileExistsError:
            continue
        return run_id, run_directory


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = REQUIRED_MANIFEST_FIELDS - manifest.keys()
    if missing:
        raise ValueError(f"Manifest is missing required fields: {sorted(missing)}")
    counts = manifest["actual_case_counts"]
    if not isinstance(counts, Mapping) or set(counts) != {"train", "validation", "test"}:
        raise ValueError("actual_case_counts must contain exactly train, validation, and test")
    for split, count in counts.items():
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
            raise ValueError(f"actual_case_counts.{split} must be a non-negative integer or null")


def write_manifest(run_directory: Path, manifest: Mapping[str, Any]) -> Path:
    """Validate and atomically write a manifest unless the run is completed."""
    _validate_manifest(manifest)
    manifest_path = run_directory / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED":
            raise CompletedRunError(f"Completed run cannot be overwritten: {run_directory.name}")
    _write_json_atomic(manifest_path, dict(manifest))
    return manifest_path


def git_state(repository_root: Path) -> tuple[str | None, bool | None]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
    )
    if commit.returncode != 0:
        return None, None
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
    )
    return commit.stdout.strip(), None if dirty.returncode != 0 else bool(dirty.stdout)


def initialize_run(
    *,
    runs_root: Path,
    experiment_id: str,
    model: str,
    seed: int | None,
    device: str | None,
    source_airfrans_manifest: Path,
    selected_case_manifest: Path,
    resolved_config: Mapping[str, Any],
    actual_case_counts: Mapping[str, int | None],
    repository_root: Path,
) -> tuple[str, Path]:
    """Create one run directory and its initial, not-yet-trained artifacts."""
    config_hash = config_sha256(resolved_config)
    source_hash = file_sha256(source_airfrans_manifest)
    selection_hash = file_sha256(selected_case_manifest)
    commit, dirty = git_state(repository_root)
    created_at = datetime.now(UTC).isoformat()
    run_id, run_directory = create_run_directory(runs_root, experiment_id)
    manifest = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "model": model,
        "seed": seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": device,
        "source_airfrans_manifest_sha256": source_hash,
        "selected_case_manifest_sha256": selection_hash,
        "config_sha256": config_hash,
        "actual_case_counts": dict(actual_case_counts),
        "status": "CREATED",
        "failure_reason": None,
        "checkpoint_path": None,
        "created_at_utc": created_at,
    }
    write_manifest(run_directory, manifest)
    _write_json_atomic(run_directory / "resolved_config.json", dict(resolved_config))
    _write_json_atomic(run_directory / "training_history.json", [])
    _write_json_atomic(run_directory / "metrics.json", {})
    _write_json_atomic(run_directory / "per_case_metrics.json", [])
    _write_json_atomic(
        run_directory / "timing.json",
        {
            "created_at_utc": created_at,
            "started_at_utc": None,
            "completed_at_utc": None,
            "elapsed_seconds": None,
        },
    )
    return run_id, run_directory
