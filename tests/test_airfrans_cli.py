import sys

import pytest

from sciai.airfrans import cli


class StopAfterArgumentParsing(RuntimeError):
    pass


@pytest.mark.parametrize(
    ("command", "target", "expected_defaults"),
    [
        (
            cli.model_comparison,
            cli.comparison_runner,
            {
                "--summary-json": "reports/summaries/model_comparison_summary.json",
                "--summary-markdown": "reports/summaries/model_comparison_results.md",
            },
        ),
        (
            cli.model_comparison_evidence,
            cli.evidence_runner,
            {
                "--evidence-json": "reports/summaries/model_comparison_evidence_data.json",
                "--profile-json": "reports/summaries/model_comparison_profile.json",
                "--predictions": "reports/summaries/model_comparison_representative_predictions.npz",
                "--report": "reports/summaries/model_comparison_evidence.md",
                "--figure-directory": "reports/figures/model_comparison",
            },
        ),
        (
            cli.edge_feature_ablation,
            cli.ablation_runner,
            {
                "--summary-json": "reports/summaries/edge_feature_ablation_data.json",
                "--report": "reports/summaries/edge_feature_ablation_report.md",
            },
        ),
    ],
)
def test_descriptive_commands_add_output_defaults(monkeypatch, command, target, expected_defaults):
    captured_arguments = []
    monkeypatch.setattr(target, "main", lambda: captured_arguments.extend(sys.argv[1:]))
    monkeypatch.setattr(sys, "argv", ["command", "--summary-json=custom.json"])

    command()

    assert sys.argv == ["command", "--summary-json=custom.json"]
    assert captured_arguments.count("--summary-json=custom.json") == 1
    assert "--summary-json" not in captured_arguments
    for option, value in expected_defaults.items():
        if option != "--summary-json":
            option_index = captured_arguments.index(option)
            assert captured_arguments[option_index + 1] == value


def test_model_comparison_evidence_defaults_match_real_parser(monkeypatch, tmp_path):
    def stop_before_data_access(*args, **kwargs):
        raise StopAfterArgumentParsing

    monkeypatch.setattr(cli.evidence_runner, "load_sources", stop_before_data_access)
    monkeypatch.setattr(sys, "argv", ["airfrans-model-comparison-evidence", f"--repository-root={tmp_path}"])

    with pytest.raises(StopAfterArgumentParsing):
        cli.model_comparison_evidence()

    assert sys.argv == ["airfrans-model-comparison-evidence", f"--repository-root={tmp_path}"]

