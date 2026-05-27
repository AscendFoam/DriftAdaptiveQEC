# T61 Review

- Verdict: `BLOCK`
- Review basis: I inspected the live diff, `T61` task package, `T61` worker summary, `T59/T60` review context, `runs/p4_benchmark/T61_statcalib_fairness_sanity_20260527_015239/*`, and `git reflog`. I did not rerun the benchmark.

## Blocking issues

- `T61` did not actually close the provenance blocker it was created to repair. The task is not just a fairness sanity rerun; it is explicitly a `clean-provenance` fairness sanity rerun. Preflight clean-start state was `HEAD=9174065`, but the final run artifact in `runs/p4_benchmark/T61_statcalib_fairness_sanity_20260527_015239/summary.json` is anchored to `git_commit=6058f42`.
- `git reflog --date=iso --all` shows an in-flight checkout from `main` to `codex-pro-research-governance-plan` while the long run was still active. Because the benchmark records commit identity at summary-generation time, not launch time, the finished artifact no longer has a single defensible code identity.
- This is materially blocking, not cosmetic. `git diff --name-only 9174065 6058f42 -- cnn_fpga tests` is non-empty and includes benchmark/runtime/config/test paths. So the T61 run cannot honestly be called a clean-provenance rerun, and `R27` is not closed.

## Non-blocking issues

- The bounded matrix itself stayed within scope:
  - scenarios: `static_bias_theta`, `linear_ramp`
  - modes: `ukf`, `hybrid_residual_b`, `statcalib`
  - `--paired-seeds`
  - `--repeats 2`
- Only one T61-scoped run root was created: `runs/p4_benchmark/T61_statcalib_fairness_sanity_20260527_015239`.
- No source, test, source-tree config, governance, or theory-only files were modified in the diff under review.
- The result signal itself persisted: `statcalib` remained the winner in both scenarios with stable `generated` status and `statcalib_generated_windows_mean=600.0`.
- `progress.jsonl` contains duplicate `running` entries for one resumed repeat. This does not appear to corrupt the final aggregates, but it does show the resume path is operationally messy.

## Missing tests

- None relative to the literal T61 change set, because T61 did not modify source code.
- If the next prerequisite chooses to fix provenance capture in benchmark code, that follow-up should add focused regression coverage for launch-time commit capture versus end-of-run commit capture.

## Suspicious implementation details

- The key tooling weakness exposed by T61 is that `run_p4_multiscenario_benchmark.py` appears to stamp `git_commit` at the end of the run rather than preserving launch-time identity in the artifact set. That makes long or resumed runs vulnerable to unrelated branch movement.
- The fairness signal is still scientifically surprising: a deliberately minimal `statcalib` heuristic lane continues to beat both `ukf` and `hybrid_residual_b` by large margins in this tiny matrix. T61 shows the signal persisted, but it does not make the comparator formally trustworthy yet.

## Recommended next action

- Do not treat T61 as the gate that opens `FR8`.
- Open one more bounded prerequisite focused on provenance isolation, not comparator expansion. The cleanest options are:
  - rerun the same exact T61 matrix in an execution environment where branch/worktree movement cannot occur during the job
  - or, under a separate allowed source-change task, harden the benchmark runner so launch-time commit identity is captured and preserved in the run artifacts
- After that provenance-safe rerun exists, re-evaluate whether the next step is an `FR8` gate or another fairness/comparator prerequisite.
