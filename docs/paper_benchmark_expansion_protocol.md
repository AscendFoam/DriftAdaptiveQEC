# Paper Benchmark Expansion Protocol

## 1. Status And Scope

This document is a docs-only protocol lock for possible paper-grade benchmark expansion.

It does not:

- run benchmark
- run training
- run `.tflite`
- call hardware
- modify benchmark code or config
- upgrade existing frozen-set evidence into expanded-benchmark evidence

Its purpose is to decide whether the paper should remain frozen-set only, or whether a separate bounded expansion lane should be prepared first.

## 2. Current Benchmark Boundary

### 2.1 What is already real

The current strongest benchmark evidence remains:

- `T24` frozen-set formal software revalidation
- `4 scenarios x 5 modes x repeats=2`
- mock-backed software HIL only
- winner in all four scenarios: `hybrid_residual_b`

Anchor evidence paths:

- `docs/P4_benchmark_formal_protocol.md`
- `docs/review/T24_review.md`
- `docs/review/T25_p4_formal_evidence_gate_review.md`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`

### 2.2 What this boundary can honestly support

The frozen set can support:

- an honest bounded paper story
- a frozen-set ranking claim
- a software-HIL-only method comparison claim

The frozen set cannot by itself support:

- a broad superiority claim
- a paper-grade expanded benchmark claim
- a deployment/runtime/board validation claim

This remains consistent with:

- `docs/paper_claim_evidence_ledger.md` (`C2`, `C3` supported; `C11` blocked)
- `docs/paper_reviewer_risk_audit.md` (`E3`)
- `docs/reality_recovery/00_freeze_snapshot.md`

## 3. Decision Frame

T45 answers two different questions separately:

1. Can the paper stay frozen-set only and still remain honest?
2. Is frozen-set only sufficient for a stronger method-value submission story?

Protocol answer:

- `Yes` for honesty, if the paper is positioned conservatively as evidence-bounded and frozen-set only.
- `No` for stronger benchmark-driven method claims.

Therefore:

- frozen-set only is acceptable as a minimum truthful paper posture
- frozen-set only is not sufficient for a stronger paper-grade benchmark story
- any broadening must be introduced as a separate bounded expansion lane, not by rewriting T24/T25

## 4. Candidate Expansion Ledger

| Candidate item | Why it matters | T45 decision | Handling rule |
| --- | --- | --- | --- |
| Keep T24 frozen set as paper anchor table | preserves comparability and avoids rewriting known evidence | `adopted` | every later expansion task must keep T24 as the anchor result pack |
| Add extra drift families beyond the frozen four | addresses reviewer concern that the benchmark is too narrow | `adopted_in_principle` | add only in a separately labeled expansion lane |
| Add `random_walk` drift family | tests slow nonstationary behavior that the frozen set does not cover directly | `adopted` | candidate future execution task should predeclare exact parameter grid |
| Add `burst_reset` or abrupt recovery drift family | tests re-lock / recovery behavior after sharp distribution changes | `adopted` | keep separate from the frozen-set table |
| Add unseen drift-law generalization holdout | tests whether the method overfits the named scenario families | `adopted` | future task must declare holdout family before execution |
| Add `sinusoidal` as a separate required family | appears in reference suggestions, but the repo already has `periodic_drift` in the frozen set | `rejected_as_required_new_family` | only add if later execution task shows that existing `periodic_drift` is not an adequate oscillatory proxy |
| Add `statcalib` comparator | directly addresses paper-facing comparator weakness without changing current mainline truth | `adopted_as_separate_lane` | include only as a clearly labeled future comparator lane, not by rewriting T24 |
| Add soft-information / correlation-aware comparator | likely valuable against reviewer skepticism | `deferred` | not current code-path ready; needs its own feasibility/integration task first |
| Add more learned branches (`Gated v5`, FiLM-style, teacher-representation variants) to the main benchmark lane | would reopen model-search and blur the mainline benchmark story | `rejected_for_current_mainline` | do not add inside the paper-grade expansion lane unless a later task explicitly reframes the paper around branch comparison |
| Expand repeats through CI-driven stopping | improves statistical confidence | `deferred` | future task may adopt CI reporting, but T45 does not lock a new stopping rule |
| Require explicit training-seed vs evaluation-seed separation for learned modes | prevents seed ambiguity in paper claims | `adopted` | later execution task must report provenance and evaluation policy separately |
| Add latency / commit / saturation / violation metrics to the expansion pack | supports the system-constrained method story | `adopted` | treat these as required report items for any future expansion task |
| Add rollback / fallback metrics as first-class acceptance fields | useful but not yet first-class in current outputs | `deferred` | needs runner/output support before being made mandatory |
| Fold true `.tflite` runtime or `real_board` into the same benchmark expansion task | mixes benchmark broadening with deployment-boundary validation | `rejected` | keep deployment boosters in separate later tasks |

## 5. Locked Rules For Any Future Expansion Lane

If a later task widens the paper benchmark story, it must obey all of the following:

1. The T24 frozen set remains unchanged and remains separately reported.
2. Expanded results must be labeled as `expansion lane`, not as a silent redefinition of the formal frozen set.
3. Any new comparator must be named as `additional comparator`, not as if it had always been part of the historical ranking set.
4. Any new scenario family must be predeclared before execution.
5. Learned-mode provenance must distinguish:
   - artifact path
   - training config or source task
   - training seed information if known
   - evaluation seed policy
6. Expanded reporting must continue to say `mock-backed software HIL` unless later runtime/board tasks produce stronger evidence.
7. Reference documents may inspire future scope, but they do not become current truth or required completion by themselves.

## 6. Required Metrics And Evidence For A Future Expansion Task

Any future execution task that claims paper-grade benchmark broadening should report at minimum:

1. per-scenario winners and runner-up gaps
2. `final_ler_mean` and `final_ler_std`
3. raw per-repeat rows
4. `overflow_rate_mean`
5. `histogram_input_saturation_rate_mean`
6. `correction_saturation_rate_mean`
7. `aggressive_param_rate_mean`
8. `n_commits_applied_mean`
9. `slow_update_violation_rate_mean`
10. `fast_cycle_violation_rate_mean`
11. explicit missing-run accounting
12. scenario/mode coverage accounting
13. exact CLI shape
14. exact config path and `config_hash`
15. explicit note on whether execution was chunked or resumed

Additional future evidence requirements:

1. separate reporting for the frozen anchor table and the expanded table
2. separate labeling for any new comparator lane such as `statcalib`
3. explicit statement that `.tflite` and `real_board` remain out of scope unless separately validated
4. explicit note on which candidate items from this document were adopted versus still deferred

## 7. Gap Audit

### 7.1 Gaps that matter for a stronger paper benchmark story

The main remaining gaps are:

1. benchmark breadth remains limited to the frozen four-scenario set
2. no separate `statcalib` comparator evidence exists yet
3. no soft-information / correlation-aware comparator exists in the current runnable lane
4. learned-mode seed/provenance reporting is still narrower than a stronger paper package would ideally want
5. mechanism evidence is still not multi-seed closed, even if benchmark ranking exists

### 7.2 Gaps that do not belong inside this benchmark protocol

The following are important, but they are not benchmark-expansion items:

1. true `.tflite` runtime restoration
2. `real_board` HIL validation
3. full training reproducibility beyond one clean-environment smoke

These remain separate evidence lanes and should not be bundled into benchmark broadening.

## 8. Go / No-Go Recommendation

### 8.1 Recommendation for frozen-set only posture

`GO`, but only for a conservative paper posture:

- evidence-bounded
- frozen-set only
- software-HIL only
- no broad superiority wording

### 8.2 Recommendation for stronger paper-grade benchmark positioning

`NO_GO` if the project intends to rely on the current frozen set alone.

Reason:

- the current benchmark is real and useful, but too narrow to carry a stronger empirical positioning by itself

### 8.3 Recommendation for next-step work

`GO_FOR_SEPARATE_BOUNDED_EXPANSION_PROTOCOL_EXECUTION`

Meaning:

- if the paper wants a stronger benchmark story, the next task should open a separate expansion execution lane
- that lane should preserve T24 as the anchor and add only predeclared breadth
- the expansion lane should not absorb deployment-boundary tasks

## 9. Explicit Non-Claims

This protocol does not claim:

1. that expanded benchmark evidence already exists
2. that `docs/reference/延伸改进思路.md` is current mainline truth
3. that the deep-research report defines mandatory repository scope
4. that the frozen T24 set was insufficient for all paper purposes
5. that `statcalib` is already an integrated comparator
6. that `.tflite` runtime or `real_board` evidence belongs to this benchmark lane
7. that any benchmark code, config, or baseline semantics have been changed

## 10. Practical Conclusion

T45 locks the paper-facing benchmark answer as:

- keep the current frozen-set formal software revalidation as the anchor benchmark truth
- do not overread it as paper-grade expanded evidence
- if stronger empirical positioning is desired, use a separate bounded expansion lane with predeclared comparators, scenario families, and reporting rules

That is the narrowest protocol that preserves current evidence honesty while still leaving a controlled path to stronger benchmark support.
