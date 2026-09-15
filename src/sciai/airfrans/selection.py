from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .audit import build_selected_cases, write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_audit_report(
    path: Path,
    selected: dict[str, object],
    selected_file_sha256: str,
    selection_script_sha256: str,
) -> None:
    counts = selected["counts"]
    source = selected["source"]
    protocol = selected["protocol"]
    report = f"""# AirfRANS split audit

Status: FROZEN for the first experiment

CFD case files downloaded by this selection step: no

## Frozen hashes

- Source manifest: `{source["manifest"]}`
- Source manifest SHA-256: `{source["manifest_file_sha256"]}`
- Selected cases manifest: `data/manifests/airfrans_selected_cases.json`
- Selected cases manifest SHA-256: `{selected_file_sha256}`
- Selection script: `{protocol["selection_script"]}`
- Selection script version: `{protocol["selection_script_version"]}`
- Selection script SHA-256: `{selection_script_sha256}`
- Git commit: unavailable because this workspace is not a Git repository
- Random seed: `{protocol["seed"]}`

## Exact selection rule

Exact geometry key: {protocol["exact_geometry_key"]}

{protocol["selection_rule"]}

Whole geometry groups are preserved. Case IDs are stored in deterministic order, and the selected manifest contains a canonical payload hash in `selection_sha256`.

## Actual counts

| Split | Status/count |
| --- | ---: |
| Train | {counts["train"]} |
| Validation | {counts["validation"]} |
| Distribution-ID / unseen-geometry test | {counts["id_test"]} |
| Geometry-OOD | NOT_DEFINED |
| Geometry-OOD case-list length | {counts["geometry_ood_case_list_length"]} |
| Total unique selected cases | {selected["total_unique_cases"]} |
| Cases with unparsed geometry metadata | {len(selected["unparsed_geometry_metadata"])} |

## Interpretation

Every selected case has a unique exact geometry key. The 50-case test is distribution-ID under the official full/scarce interpolation task, while its exact geometries are unseen in train and validation. It is **not** a geometry-OOD test.

Geometry-OOD is `NOT_DEFINED` and remains a future experiment. Its case list is intentionally empty. A geometry holdout criterion must be approved before any geometry-OOD IDs are selected.

This audit freezes IDs and metadata only. It is not a CFD data audit, model test, or training result.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main(selection_script: Path | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/airfrans_hf/data/Dataset/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/airfrans_selected_cases.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/airfrans_split_audit.md"),
    )
    args = parser.parse_args()

    source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = build_selected_cases(source_manifest, source_file_sha256=file_sha256(args.manifest))
    write_json(args.output, selected)
    selected_file_sha256 = file_sha256(args.output)
    selection_script_sha256 = file_sha256(selection_script or Path(__file__))
    write_audit_report(args.report, selected, selected_file_sha256, selection_script_sha256)
    print(
        json.dumps(
            {
                "counts": selected["counts"],
                "total_unique_cases": selected["total_unique_cases"],
                "source_manifest_sha256": selected["source"]["manifest_file_sha256"],
                "selected_cases_sha256": selected_file_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
