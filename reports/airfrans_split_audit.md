# AirfRANS split audit

Status: FROZEN for the first experiment

CFD case files downloaded by this selection step: no

## Frozen hashes

- Source manifest: `data/airfrans_hf/data/Dataset/manifest.json`
- Source manifest SHA-256: `3d1320005bcf3d94df80df2bf8338dd6ce6d394c6beafa14849b4ce1de7def13`
- Selected cases manifest: `data/manifests/airfrans_selected_cases.json`
- Selected cases manifest SHA-256: `8c532a15c2af6538177ae9885de8978553b162c4eb788d33a2998c9edc447014`
- Selection script: `scripts/select_airfrans_cases.py`
- Selection script version: `airfrans-selection-v1`
- Selection script SHA-256: `4f7ca0b4b7f912f2152c6f63ef7c495dfba702de38912686a15e89dd1299221f`
- Git commit: unavailable because this workspace is not a Git repository
- Random seed: `airfrans-m1-v1`

## Exact selection rule

Exact geometry key: All underscore-delimited case ID tokens from index 4 onward.

Use all 200 official scarce_train cases. Keep the first sorted case's exact geometry group in train. Rank every other exact geometry group by SHA-256(seed + ':validation:' + geometry_key), then add whole groups when doing so reduces the distance from the validation target of 40; assign the rest to train. From official full_test, retain cases whose exact geometry key is absent from the 200-case pool, rank by SHA-256(seed + ':test:' + case_id), take the first 50, then sort IDs for storage.

Whole geometry groups are preserved. Case IDs are stored in deterministic order, and the selected manifest contains a canonical payload hash in `selection_sha256`.

## Actual counts

| Split | Status/count |
| --- | ---: |
| Train | 160 |
| Validation | 40 |
| Distribution-ID / unseen-geometry test | 50 |
| Geometry-OOD | NOT_DEFINED |
| Geometry-OOD case-list length | 0 |
| Total unique selected cases | 250 |
| Cases with unparsed geometry metadata | 0 |

## Interpretation

Every selected case has a unique exact geometry key. The 50-case test is distribution-ID under the official full/scarce interpolation task, while its exact geometries are unseen in train and validation. It is **not** a geometry-OOD test.

Geometry-OOD is `NOT_DEFINED` and remains a future experiment. Its case list is intentionally empty. A geometry holdout criterion must be approved before any geometry-OOD IDs are selected.

This audit freezes IDs and metadata only. It is not a CFD data audit, model test, or training result.
