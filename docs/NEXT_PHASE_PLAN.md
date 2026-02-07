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

