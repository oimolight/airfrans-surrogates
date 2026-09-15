from __future__ import annotations

import sys
from collections.abc import Callable, Mapping

from . import edge_feature_ablation as ablation_runner
from . import evidence as evidence_runner
from . import model_comparison as comparison_runner


def _run_with_defaults(
    main: Callable[[], None],
    defaults: Mapping[str, str],
) -> None:
    original_argv = sys.argv[:]
    arguments = sys.argv[1:]
    for option, value in defaults.items():
        if not any(argument == option or argument.startswith(f"{option}=") for argument in arguments):
            sys.argv.extend((option, value))
    try:
        main()
    finally:
        sys.argv[:] = original_argv


def model_comparison() -> None:
    _run_with_defaults(
        comparison_runner.main,
        {
            "--summary-json": "reports/summaries/model_comparison_summary.json",
            "--summary-markdown": "reports/summaries/model_comparison_results.md",
        },
    )


def model_comparison_evidence() -> None:
    _run_with_defaults(
        evidence_runner.main,
        {
            "--evidence-json": "reports/summaries/model_comparison_evidence_data.json",
            "--profile-json": "reports/summaries/model_comparison_profile.json",
            "--predictions": "reports/summaries/model_comparison_representative_predictions.npz",
            "--report": "reports/summaries/model_comparison_evidence.md",
            "--figure-directory": "reports/figures/model_comparison",
        },
    )


def edge_feature_ablation() -> None:
    _run_with_defaults(
        ablation_runner.main,
        {
            "--summary-json": "reports/summaries/edge_feature_ablation_data.json",
            "--report": "reports/summaries/edge_feature_ablation_report.md",
        },
    )
