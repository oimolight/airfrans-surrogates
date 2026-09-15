# AirfRANS selective download preflight

Status: AWAITING APPROVAL

CFD case files downloaded by this step: no

## Source and destination

- Hugging Face dataset: `OneScience-Group/airfrans`
- Pinned revision: `f99b37789d0c76e02b923b14b23dcd6389f0d895`
- Frozen selection: `data/manifests/airfrans_selected_cases.json`
- Frozen selection SHA-256: `8c532a15c2af6538177ae9885de8978553b162c4eb788d33a2998c9edc447014`
- Destination: `data/raw/airfrans_hf_selected`

## Planned files

- Selected unique cases: 250
- Files per case: 3 (`_aerofoil.vtp`, `_freestream.vtp`, `_internal.vtu`)
- Selected case files: 750
- Source manifest files: 1
- Total files: 751
- Unselected case files excluded: 2250
- Missing selected cases: 0
- Unknown remote sizes: 0

## Storage

- Expected case bytes: 3745395822
- Expected source manifest bytes: 187191
- Expected total bytes: 3745583013
- Available bytes before download: 52866842624
- Expected available bytes after download: 49121259611

## Path safety

Every planned remote path is relative, contains neither `.` nor `..`, begins with `data/Dataset/`, and resolves inside the destination. Case files are selected only when the immediate parent directory exactly matches a frozen case ID. No substring or wildcard case matching is used.

The preserved ISIR partial is outside the destination and is not deleted or read by the planned download.
