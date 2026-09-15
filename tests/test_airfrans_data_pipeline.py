import copy
import tempfile
from pathlib import Path, PurePosixPath

import pytest

from sciai.airfrans.audit import build_selected_cases, build_split, canonical_hash, case_metadata
from sciai.airfrans.hf_download import verify_download
from sciai.airfrans.hf_preflight import safe_destination_path, safe_relative_path


def make_case(index: int, geometry: int, speed: float = 50.0, angle: float = 0.0) -> str:
    return f"airFoil2D_SST_{speed:.3f}_{angle:.3f}_{geometry:.3f}_2.000_12.000"


class TestAirfransSplit:
    def setup_method(self) -> None:
        scarce = [make_case(index, index) for index in range(200)]
        official_test = [make_case(index, index + 1000, speed=60.0) for index in range(200)]
        self.manifest = {
            "scarce_train": scarce,
            "full_test": official_test,
            "full_train": list(reversed(scarce)),
        }
        self.audit_case = sorted(scarce)[0]

    def test_split_counts_overlap_and_audit_case(self) -> None:
        split = build_split(self.manifest, self.audit_case)
        train = set(split["local_train"])
        validation = set(split["local_validation"])
        test = set(split["local_test"])
        assert split["actual_counts"] == {"train": 160, "validation": 40, "test": 50}
        assert not train & validation
        assert not train & test
        assert not validation & test
        assert self.audit_case in train
        assert self.audit_case not in test
        assert split["test_sealed"]
        assert split["calibration"]["status"] == "not_applicable_for_airfrans_spec"

    def test_geometry_groups_do_not_overlap(self) -> None:
        split = build_split(self.manifest, self.audit_case)
        groups = {
            role: {case_metadata(case_id)["geometry_group"] for case_id in split[role]}
            for role in ("local_train", "local_validation", "local_test")
        }
        assert not groups["local_train"] & groups["local_validation"]
        assert not groups["local_train"] & groups["local_test"]
        assert not groups["local_validation"] & groups["local_test"]

    def test_manifest_order_does_not_change_split_or_hash(self) -> None:
        reversed_manifest = {key: list(reversed(value)) for key, value in self.manifest.items()}
        assert build_split(self.manifest, self.audit_case) == build_split(
            reversed_manifest, self.audit_case
        )

    def test_target_values_cannot_affect_split(self) -> None:
        manifest_with_unrelated_targets = copy.deepcopy(self.manifest)
        first = build_split(manifest_with_unrelated_targets, self.audit_case)
        unrelated_targets = {case_id: index * 1000 for index, case_id in enumerate(self.manifest["scarce_train"])}
        unrelated_targets = {key: -value for key, value in unrelated_targets.items()}
        second = build_split(manifest_with_unrelated_targets, self.audit_case)
        assert first == second

    def test_hash_is_canonical(self) -> None:
        assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})

    def test_selected_cases_keep_geometry_ood_unselected(self) -> None:
        selected = build_selected_cases(self.manifest)
        assert selected["counts"] == {
            "train": 160,
            "validation": 40,
            "id_test": 50,
            "geometry_ood_test": "NOT_DEFINED",
            "geometry_ood_case_list_length": 0,
        }
        assert selected["splits"]["geometry_ood_test"] == []
        assert selected["protocol"]["geometry_ood_status"] == "NOT_DEFINED"
        assert selected["unparsed_geometry_metadata"] == []
        assert selected["total_unique_cases"] == 250


class TestAirfransHfData:
    def test_safe_paths_remain_inside_destination(self) -> None:
        destination = Path("data/raw/airfrans_hf_selected")
        valid = safe_relative_path("data/Dataset/case/case_internal.vtu")
        resolved = safe_destination_path(destination, valid)
        assert resolved.is_relative_to(destination.resolve())

    def test_unsafe_paths_are_rejected(self) -> None:
        destination = Path("data/raw/airfrans_hf_selected")
        for path in ("/tmp/x", "../x", "data/Dataset/../../x", "data/Dataset/./x", "data//Dataset/x"):
            with pytest.raises(RuntimeError):
                safe_relative_path(path)
        with pytest.raises(RuntimeError):
            safe_destination_path(destination, PurePosixPath("data/Dataset/../../x"))

    def test_verify_download_rejects_unplanned_case_file(self) -> None:
        destination = self.directory / "selected"
        case_path = destination / "data" / "Dataset" / "case" / "case_internal.vtu"
        case_path.parent.mkdir(parents=True)
        case_path.write_bytes(b"valid")
        plan = {
            "destination": str(destination),
            "files": [{"remote_path": "data/Dataset/case/case_internal.vtu", "size_bytes": 5}],
        }
        assert verify_download(plan) == (1, 5)

        extra_path = destination / "data" / "Dataset" / "other" / "other_internal.vtu"
        extra_path.parent.mkdir(parents=True)
        extra_path.write_bytes(b"extra")
        with pytest.raises(RuntimeError, match="Downloaded path set differs"):
            verify_download(plan)

    def setup_method(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def teardown_method(self) -> None:
        self.temporary_directory.cleanup()
