# FR8 Statcalib Comparator Gate Review

## Verdict

- Recommendation: `GO_FOR_BOUNDED_FR8_TASK`
- `T63` is a docs-only gate review. It is not `FR8` evidence.
- Current evidence remains `mock`-backed software-HIL only. It does not validate `.tflite` runtime behavior or real-board behavior.
- `R27` is closed by `T62`.
- `R24` is not closed today, but it no longer needs another pre-gate task. It becomes the main scope constraint for the next bounded `FR8` extension-lane task.

## Evidence Status

| Item | Status | Local evidence | Gate reading |
| --- | --- | --- | --- |
| Separate-lane `statcalib` concept and interface contract exist | Closed | `docs/review/T26_review.md`, `docs/review/T30_review.md` | `T26` accepted the lane as a separate later comparator concept. `T30` then locked the interface contract and minimal estimator semantics without claiming integrated benchmark evidence. |
| Separate-lane benchmark integration exists | Closed | `docs/review/T59_review.md`, `docs/evidence_packs/statcalib_fr8/statcalib_comparator_lane_smoke.md`, `runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740/summary.json` | `T59` proved that `statcalib` can run as a distinct benchmark lane with propagated `statcalib_status` / `statcalib_reason` fields. |
| Cross-mode fallback leakage is closed | Closed | `docs/review/T60_review.md`, `docs/evidence_packs/statcalib_fr8/statcalib_lane_isolation_and_regression_hardening.md` | `T60` closed the `teacher_mode` leakage blocker and added regression coverage around lane isolation. |
| Direct regression hardening exists | Closed | `docs/review/T60_review.md`, `docs/evidence_packs/statcalib_fr8/statcalib_lane_isolation_and_regression_hardening.md` | `T60` added estimator-branch coverage and aggregation/report regression tests. |
| Provenance-clean bounded fairness sanity evidence exists | Closed | `docs/review/T62_review.md`, `docs/evidence_packs/statcalib_fr8/statcalib_provenance_isolated_fairness_rerun.md`, `runs/p4_benchmark/T62_statcalib_provenance_isolated_20260527_122943/summary.json` | `T62` reran the bounded smoke matrix on clean `main`; launch / finish / `summary.json` commit identity all match `e2773d3`, with `missing_runs_count=0`, `coverage=1.0`, and `completed_repeats=2` for all rows. |
| Broader fairness / robustness outside the tiny two-scenario matrix | Open | `docs/evidence_packs/statcalib_fr8/statcalib_comparator_lane_smoke.md`, `docs/evidence_packs/statcalib_fr8/statcalib_fairness_sanity.md`, `docs/evidence_packs/statcalib_fr8/statcalib_provenance_isolated_fairness_rerun.md`, `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` | `T59`, `T61`, and `T62` only cover `static_bias_theta` and `linear_ramp`. The locked four-scenario formal boundary also includes `step_sigma_theta` and `periodic_drift`. |
| Defensible formal-comparator positioning for the minimal heuristic lane | Open | `docs/review/T26_review.md`, `docs/review/T30_review.md`, `docs/paper_materials/paper_claim_evidence_ledger.md`, `docs/paper_materials/paper_ablation_result_pack.md` | The repository still does not have a full extension-lane result pack showing how `statcalib` behaves against the frozen comparison set. Current evidence is enough for admission to a bounded extension task, not enough for a completed formal-comparator claim. |
| Any evidence beyond mock-backed software-HIL | Open | `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`, `docs/04_task_board.md`, `docs/07_handoff.md`, `runs/p4_benchmark/T62_statcalib_provenance_isolated_20260527_122943/summary.json` | The protocol and latest Captain state both keep `.tflite` and real-board validation outside this lane, and the bounded rerun still records `backend=mock`. |

## R27 Closure Decision

Decision: `R27` should now be treated as closed by `T62`.

Reason:

1. `docs/04_task_board.md` states that `T62` closed the specific blocker that caused `T61` to fail and that `R27` should now be treated as closed.
2. `docs/07_handoff.md` repeats the same closure decision and makes `T63` the next gate task rather than another provenance-repair task.
3. `docs/review/T62_review.md` confirms that the bounded rerun stayed on clean `main`, kept one commit identity from launch through `summary.json`, and produced no duplicate `running` noise.
4. `runs/p4_benchmark/T62_statcalib_provenance_isolated_20260527_122943/summary.json` matches that audit result: six comparison rows, zero missing runs, and full repeat coverage.

This closes the clean-provenance blocker from `T61`. It does not close `R24`, and it does not upgrade the evidence into finished `FR8`.

## Why The Gate Result Is GO

The remaining gaps are exactly the gaps that a bounded `FR8` extension-lane task should answer:

1. wider scenario coverage inside the already locked four-scenario software benchmark boundary
2. a same-protocol comparison between `statcalib` and the frozen comparison set
3. a clean, separately labeled result pack that does not silently rewrite the frozen ranked table

Another pre-`FR8` prerequisite is not the smallest honest next step anymore because:

1. semantics isolation is already closed by `T60`
2. provenance isolation is already closed by `T62`
3. the formal protocol already defines how `statcalib` may enter later work: as a separately labeled extension lane, not as a silent frozen-set rewrite (`docs/protocols/benchmark/P4_benchmark_formal_protocol.md`, calibration/statcalib rule)

The safest next move is therefore one bounded `FR8` task, not one more abstract gate loop.

## Smallest Safe FR8 Scope

The next task should be a bounded `FR8` extension-lane task with all of the following constraints:

1. Use `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark`.
2. Use `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` with no semantic config edits.
3. Keep the locked four-scenario set:
   - `static_bias_theta`
   - `linear_ramp`
   - `step_sigma_theta`
   - `periodic_drift`
4. Keep the frozen five-mode ranked table unchanged:
   - `ekf`
   - `ukf`
   - `constant_residual_mu`
   - `rls_residual_b`
   - `hybrid_residual_b`
5. Add `statcalib` only as one separately labeled extension lane. Do not merge it into the frozen ranked table.
6. Keep `--paired-seeds` and `--repeats 2`.
7. If chunking is needed, chunk only by repeat range so the full scenario order and seed semantics stay intact, matching the formal protocol.
8. Produce one clean-provenance run root and require launch / finish / `summary.json` commit identity to match.
9. Report, at minimum:
   - per-scenario winners and runner-up gaps
   - `final_ler_mean` and `final_ler_std`
   - coverage and `missing_runs`
   - raw per-repeat rows
   - `statcalib_status`, `statcalib_reason`, and generated-window counts
10. Keep the explicit boundary statement in every result doc:
   - mock-backed software-HIL only
   - not `.tflite`
   - not real-board
   - not paper-grade expanded benchmark evidence

This is the smallest safe `FR8` scope because `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` already records that adding one extra mode across the frozen four-scenario, two-repeat set is a `+8 repeat-run` increment. That is materially smaller and safer than expanding scenarios, expanding repeats, or mixing deployment-boundary validation into the same task.
