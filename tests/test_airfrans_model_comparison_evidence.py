import pytest
from sciai.airfrans.evidence import (
    METRICS,
    MODEL_NAMES,
    SEEDS,
    build_aggregate_metrics,
    build_paired_comparisons,
    distribution_summary,
    paired_case_summary,
    select_representative_cases,
    source_key,
)


class TestModelComparisonEvidence:
    @staticmethod
    def sources() -> dict[str, dict[str, object]]:
        model_values = {
            "mlp": ([1.0, 100.0, 3.0], [4.0, 5.0, 6.0]),
            "gnn": ([2.0, 2.0, 2.0], [7.0, 7.0, 7.0]),
            "fno": ([8.0, 8.0, 8.0], [9.0, 9.0, 9.0]),
        }
        sources: dict[str, dict[str, object]] = {}
        for model_name in MODEL_NAMES:
            for seed_index, seed in enumerate(SEEDS):
                first_case, second_case = model_values[model_name]
                records = []
                for case_id, value in (("case-a", first_case[seed_index]), ("case-b", second_case[seed_index])):
                    record: dict[str, object] = {"case_id": case_id}
                    record.update({metric: value for metric in METRICS})
                    records.append(record)
                sources[source_key(model_name, seed)] = {
                    "seed": seed,
                    "test_records": records,
                }
        return sources

    def test_distribution_summary_uses_linear_p90(self) -> None:
        summary = distribution_summary([1.0, 2.0, 3.0, 4.0])
        assert summary["count"] == 4
        assert summary["mean"] == 2.5
        assert summary["median"] == 2.5
        assert summary["p90"] == pytest.approx(3.7)

    def test_paired_summary_aligns_case_ids(self) -> None:
        summary = paired_case_summary(
            {"case-b": 1.0, "case-a": 3.0},
            {"case-a": 2.0, "case-b": 1.5},
        )
        assert summary["first_lower"] == 1
        assert summary["second_lower"] == 1
        assert summary["ties"] == 0

    def test_model_aggregate_uses_per_case_seed_median(self) -> None:
        rows = build_aggregate_metrics(self.sources())
        row = next(
            row
            for row in rows
            if row["model"] == "mlp"
            and row["scope"] == "case_median_across_seeds"
            and row["metric"] == METRICS[0]
        )
        assert row["count"] == 2
        assert row["mean"] == 4.0

    def test_paired_model_aggregate_uses_per_case_seed_median(self) -> None:
        rows = build_paired_comparisons(self.sources())
        row = next(
            row
            for row in rows
            if row["first_model"] == "mlp"
            and row["second_model"] == "gnn"
            and row["scope"] == "case_median_across_seeds"
            and row["metric"] == METRICS[0]
        )
        assert row["count"] == 2
        assert row["mean"] == -0.5
        assert row["first_lower"] == 1
        assert row["second_lower"] == 1

    def test_representative_rule_has_deterministic_ties(self) -> None:
        selected = select_representative_cases(
            {
                "case-c": [5.0, 7.0],
                "case-b": [3.0, 5.0],
                "case-a": [1.0, 3.0],
                "case-d": [5.0, 7.0],
            }
        )
        assert selected["median_difficulty"]["case_id"] == "case-b"
        assert selected["high_error"]["case_id"] == "case-c"
