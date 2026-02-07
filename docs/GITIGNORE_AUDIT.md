# Gitignore Audit

Last updated: 2026-02-07

## Scope
- Build outputs (`bin/`, `obj/`, `out/`, `TestResults/`)
- Runtime outputs (`output/`, `outputs/`, `inference_results/`)
- Local generated artifacts (tiny infer/export outputs)
- Local control files (`AGENTS.md`)

## Findings
- Current `.gitignore` excludes generated files and model artifacts by default.
- Tracked assets/config templates are not blocked by ignore rules.
- Local run products under `assets/configs/local/output/` and `infer_*` are excluded as intended.

## Recommendation
- Keep ignore rules narrow to generated/runtime files only.
- Any new sample assets should be added under explicit tracked paths and verified via `git status`.

