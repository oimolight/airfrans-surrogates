# AirfRANS scientific ML workspace

Read README.md, docs/data_contract.md, docs/experiment_protocol.md,
docs/REPRODUCIBILITY.md and docs/PROGRESS.md before changing experiment code.
This repository contains a completed model comparison and edge-feature-ablation evidence.
Use `uv run --frozen` for every Python command; never invoke bare `python`,
`pytest` or `pip`. Use `sciai-check` for the fast repository check.

## Teach while implementing
The learner is new to scientific ML. Explain new terms on first use.
Use the sequence: purpose, small example, input/output, implementation, verification, limitation.
Prefer English explanations and short, readable functions. Do not conceal essential numerical details.
After a milestone ask one comprehension question and stop for review.

## Permission and scope
A prompt is not a security control. Respect the host's permission settings and organisation policy.
Start with a plan; implement only the explicitly approved milestone.
Ask before downloads, installations, long/paid compute, external uploads, pushes, destructive operations or publication.
Never change settings to bypass approval. This repository is AirfRANS-only.
Do not silently change data, split, model comparison or budget when a run fails.

## Evidence
Never invent real-data schema, results, timings, successful tests or approval.
Separate docs validation, synthetic fixture tests, real-data smoke runs and full experiments.
Record actual commands, outputs, failures, case IDs, software versions, config and split hashes.
Do not pass tests by deleting assertions without a justified, reviewed specification change.
Only fit label-dependent normalisers on permitted training labels.
Keep geometry/case groups disjoint and test/calibration labels out of model selection and acquisition.
A low training loss is not a generalisation result.

## Writing and safety
Label hypothetical examples.
Do not put CVs, client/internal architecture documents, secrets or interview motivations into code or publication drafts.
.gitignore is not a Copilot-access or data-handling boundary.
Keep measured findings separate from hypotheses and sourced background.
Do not mark human approval as complete or publish externally.

## Report and pause
After each approved action report changed files, actual commands, actual outcomes,
not-run checks, remaining risks and one next proposed task.
Update docs/PROGRESS.md from actual evidence, leaving human approval to the user.
Stop rather than starting the next milestone automatically.
