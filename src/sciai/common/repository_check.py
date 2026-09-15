"""Validate the shareable repository structure without running experiments."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

REQUIRED_FILES = (
    ".gitignore",
    ".tool-versions",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "configs/airfrans/model_comparison.json",
    "configs/airfrans/edge_feature_ablation.json",
    "data/manifests/airfrans_selected_cases.json",
    "docs/PROGRESS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/data_contract.md",
    "docs/experiment_protocol.md",
    "reports/summaries/edge_feature_ablation_report.md",
    "reports/summaries/model_comparison_evidence.md",
)

REQUIRED_IGNORE_RULES = (
    ".history/",
    ".venv/",
    "/runs/",
    "data/airfrans_hf/",
    "data/processed/",
    "data/raw/",
    "/reports/runs/*",
    "!/reports/runs/README.md",
    "*.pt",
)
FORBIDDEN_IGNORE_RULES = ("runs/",)

LOCAL_JSON_EXCLUSIONS = frozenset({".git", ".history", ".venv", "airfrans_hf", "raw"})
MARKDOWN_LINK = re.compile(r"!?\[[^]]*]\(([^)]+)\)")


def _candidate_json_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*.json")
        if not any(part in LOCAL_JSON_EXCLUSIONS for part in path.relative_to(root).parts)
    ]


def _broken_local_links(path: Path, root: Path) -> list[str]:
    broken: list[str] = []
    text = path.read_text(encoding="utf-8")
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
            broken.append(raw_target)
    return broken


def _gitignore_errors(text: str) -> list[str]:
    rules = {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    errors = [
        f"Missing .gitignore rule: {rule}"
        for rule in REQUIRED_IGNORE_RULES
        if rule not in rules
    ]
    errors.extend(
        f"Overbroad .gitignore rule: {rule}"
        for rule in FORBIDDEN_IGNORE_RULES
        if rule in rules
    )
    return errors


def collect_errors(root: Path = ROOT) -> list[str]:
    errors: list[str] = []

    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            errors.append(f"Missing required file: {relative}")

    for path in _candidate_json_files(root):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"Invalid JSON: {path.relative_to(root)}: {exc}")

    gitignore_path = root / ".gitignore"
    if gitignore_path.is_file():
        errors.extend(_gitignore_errors(gitignore_path.read_text(encoding="utf-8")))

    for relative in ("README.md", "docs/REPRODUCIBILITY.md"):
        path = root / relative
        if not path.is_file():
            continue
        for target in _broken_local_links(path, root):
            errors.append(f"Broken link in {relative}: {target}")

    return errors


def main(root: Path = ROOT) -> int:
    errors = collect_errors(root)
    if errors:
        print("REPOSITORY CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("REPOSITORY CHECK PASSED")
    print("Checked required files, JSON, local links, and local-artifact ignore rules.")
    print("No dataset download, model execution, scientific test, or publication was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
