# Native Implementation Checklist (20 Items)

Last updated: 2026-02-07

## Completion Definition

Each item is considered complete only when all conditions are true:
- Native `.NET 10` implementation exists (no Python dependency).
- CLI command is runnable with deterministic outputs.
- Input validation and actionable errors are present.
- Build passes (`dotnet build`), tests pass (`dotnet test`).
- At least one smoke path is demonstrated for the feature area.

## Ordered Execution

- [x] 1. Build command-to-command gap report from `tools/` to `.NET`.
- [x] 2. Define and freeze completion criteria for all feature deliveries.
- [x] 3. Design `infer table` ONNX I/O contract.
- [x] 4. Implement `infer table` end-to-end native execution.
- [x] 5. Design `infer kie` input/output model.
- [x] 6. Implement `infer kie` native execution.
- [x] 7. Implement `infer kie-ser` native execution.
- [x] 8. Implement `infer kie-re` native execution.
- [ ] 9. Upgrade rec training metrics (char acc, edit distance).
- [ ] 10. Unify export manifest schema across all model types.
- [ ] 11. Extend convert subcommand family with consistent validation.
- [ ] 12. Add service test concurrency/timeout controls.
- [ ] 13. Standardize infer output directory and naming conventions.
- [ ] 14. Add pre/post-processing component registry architecture.
- [ ] 15. Complete config view types for `table/kie/sr`.
- [ ] 16. Add tiny sample assets for `table/kie/sr`.
- [ ] 17. Add CLI contract tests for missing/invalid args.
- [ ] 18. Add deterministic output consistency tests.
- [ ] 19. Finalize project-focused `.gitignore` guardrails.
- [ ] 20. Publish final implementation matrix and phase handoff report.
