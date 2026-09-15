import json
import tempfile
from pathlib import Path

import pytest

from sciai.common.artifacts import (
    RUN_ARTIFACT_FILES,
    REQUIRED_MANIFEST_FIELDS,
    CompletedRunError,
    config_sha256,
    create_run_directory,
    initialize_run,
    write_manifest,
)


class TestArtifactContract:
    def test_config_hash_is_deterministic(self) -> None:
        first = {"model": {"width": 64, "layers": 2}, "seed": 7}
        reordered = {"seed": 7, "model": {"layers": 2, "width": 64}}
        assert config_sha256(first) == config_sha256(reordered)
        assert config_sha256(first) != config_sha256({**first, "seed": 8})

    def test_run_ids_are_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_id, first_path = create_run_directory(root, "model-comparison-mlp")
            second_id, second_path = create_run_directory(root, "model-comparison-mlp")
        assert first_id != second_id
        assert first_path != second_path

    def test_manifest_requires_every_contract_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with pytest.raises(ValueError, match="missing required fields"):
                write_manifest(Path(temporary), {"run_id": "incomplete"})

    def test_initialize_run_writes_contract_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_manifest = root / "source.json"
            selected_manifest = root / "selected.json"
            source_manifest.write_text('{"source":"fixture"}\n', encoding="utf-8")
            selected_manifest.write_text('{"selection":"fixture"}\n', encoding="utf-8")
            run_id, run_directory = initialize_run(
                runs_root=root / "runs",
                experiment_id="model-comparison-mlp",
                model="pointwise_mlp",
                seed=7,
                device="cpu",
                source_airfrans_manifest=source_manifest,
                selected_case_manifest=selected_manifest,
                resolved_config={"seed": 7},
                actual_case_counts={"train": 2, "validation": 1, "test": 1},
                repository_root=root,
            )
            manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
            assert set(RUN_ARTIFACT_FILES) == {path.name for path in run_directory.iterdir()}
        assert manifest["run_id"] == run_id
        assert REQUIRED_MANIFEST_FIELDS <= manifest.keys()
        assert manifest["status"] == "CREATED"
        assert manifest["failure_reason"] is None
        assert manifest["checkpoint_path"] is None
        assert manifest["git_commit"] is None
        assert manifest["git_dirty"] is None

    def test_completed_run_manifest_cannot_be_overwritten(self) -> None:
        manifest = {field: None for field in REQUIRED_MANIFEST_FIELDS}
        manifest.update(
            {
                "run_id": "completed-run",
                "experiment_id": "model-comparison-mlp",
                "model": "pointwise_mlp",
                "actual_case_counts": {"train": 2, "validation": 1, "test": 1},
                "status": "COMPLETED",
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            manifest_path = write_manifest(run_directory, manifest)
            original = manifest_path.read_bytes()
            with pytest.raises(CompletedRunError):
                write_manifest(run_directory, {**manifest, "status": "FAILED"})
            assert manifest_path.read_bytes() == original
