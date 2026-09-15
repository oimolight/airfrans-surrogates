import pytest
from pathlib import Path

from sciai.airfrans.model_comparison import (
    aggregate_case_metrics,
    locked_source_hashes_match,
    read_json,
    resolved_config,
    validate_case_records,
)


class TestModelComparison:
    repository_root = Path(__file__).resolve().parents[1]

    def test_historical_source_hashes_resolve_from_provenance(self) -> None:
        lock = read_json(
            self.repository_root / "reports/summaries/model_comparison_checkpoint_lock.json"
        )
        assert locked_source_hashes_match(
            self.repository_root,
            lock["evaluation_source_sha256"],
        )

    def test_macro_metrics_weight_cases_equally(self) -> None:
        aggregate = aggregate_case_metrics(
            [
                {"pressure_rmse_nondimensional": 1.0, "surface_mae_nondimensional": 2.0},
                {"pressure_rmse_nondimensional": 3.0, "surface_mae_nondimensional": 6.0},
            ]
        )
        assert aggregate["pressure_rmse_nondimensional_macro_mean"] == 2.0
        assert aggregate["surface_mae_nondimensional_macro_mean"] == 4.0

    def test_resolved_fit_config_excludes_test_case_ids(self) -> None:
        base = {
            "schema_version": 1,
            "purpose": "test",
            "models": {"mlp": {"hidden_width": 4, "hidden_layers": 2}},
            "device": "cpu",
            "sampling": {},
            "evaluation": {},
            "training": {},
            "split": {},
        }
        config = resolved_config(
            base,
            "mlp",
            17,
            {"train": ["train"], "validation": ["validation"], "id_test": ["secret-test"]},
        )
        assert "secret-test" not in str(config)
        assert config["split"]["test_case_ids_locked_after_all_fits"]

    def test_case_records_must_match_frozen_split_exactly(self) -> None:
        records = [
            {"case_id": "case-a", "split": "validation"},
            {"case_id": "case-b", "split": "validation"},
        ]
        validate_case_records(records, "validation", ["case-b", "case-a"])
        with pytest.raises(RuntimeError, match="Duplicate case ID"):
            validate_case_records([records[0], records[0]], "validation", ["case-a", "case-b"])
        with pytest.raises(RuntimeError, match="frozen validation"):
            validate_case_records(records, "validation", ["case-a", "case-c"])
