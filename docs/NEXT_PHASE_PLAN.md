# Next Phase Plan

Last updated: 2026-02-07

## Priority Focus
1. Raise `table/kie` functional parity from partial to production-ready.
2. Expand deterministic regression suites for infer/export pipelines.
3. Add benchmark harness for latency/throughput baselines per command.
4. Improve training model quality path (scheduler choices, richer metrics, dataset tooling).
5. Introduce plugin packaging conventions for preprocess/postprocess extensions.

## Acceptance Gate
- All new features must pass build + tests + smoke commands.
- Every command change requires at least one contract test and one regression assertion.
- Manifest and report schemas must remain backward-compatible (`1.x`).

## Progress Snapshot
- done: Added native `benchmark run` command with warmup/iterations/p95 throughput report (`benchmark_results/*.json`).
- done: Added plugin package convention validator via `plugin validate-package --package_dir <dir>` with `plugin.json` schema checks.
- done: Added CLI routing/help/tests for `benchmark` and `plugin` commands.
- done: Expanded regression tests to cover benchmark executor and plugin package validator.
- done: Added deterministic table output serialization regression test (`TableResultSerializerTests`).
- done: Strengthened manifest compatibility validation (`ManifestSemVer` 1.x + required compatibility fields) and regression tests.
- done: Added CI-fast KIE/table smoke assets and script (`scripts/smoke-ci-fast.ps1`).
- done: Added benchmark profile presets for `service:test` (`smoke|balanced|stress`) with option override support.
