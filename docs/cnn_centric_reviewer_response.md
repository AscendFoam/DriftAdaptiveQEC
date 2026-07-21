# T7.3.2：CNN-centric / simulator-overfitting 审稿风险回答

- verdict：`PASS_CNN_NONCENTRIC_REPLACEABLE_LEARNING_REVIEWER_RESPONSE`
- gates：`23/23`
- semantic mutations：`23/23`
- package readiness：`draft_with_placeholders`

## Response strategy summary

- Decision type: unclear; this is a pre-emptive reviewer-risk package, not a supplied decision letter.
- Overall posture: accept the framing concern, disclose favorable legacy learning evidence, separate task signatures, and retain a strict future promotion gate.
- Major risk: either hiding positive T4.4 evidence or migrating it into the current multimode/RTL task would be misleading.
- Suggested ordering: current paper center -> Phase-6D disposition -> legacy evidence -> nonmigration -> future gate.

## Comment-response tracker

| ID | Reviewer concern | Type | Severity | Proposed action | Missing author input |
| --- | --- | --- | --- | --- | --- |
| PRQ-CNN-1 | Is the CNN merely overfitting the simulator, and is the project still CNN-centric? | evidence / methodology / positioning | major | CLARIFY_EXISTING + SOFTEN_CLAIM + PARTIAL | Actual reviewer ID and verbatim wording |

## Draft point-by-point response letter

> **Placeholder:** replace `PRQ-CNN-1` with the actual reviewer ID and paste the verbatim reviewer wording before submission.

We agree that the earlier CNN--FPGA framing could make the learned component appear more central than the evidence supports. The revised manuscript is not CNN-centric: its two primary evidence lanes are (i) a strongest-baseline-gated multimode software qualification and (ii) an exact single-mode deterministic, atomic and fail-closed RTL qualification. CNN, teacher and student modules have no independent vote in either lane.

For Phase 6D, the multimode causal-headroom entry test stopped with the proposed method tied to static-mixture exact MLD, at p_L=0.1119791667 for both methods and a paired relative improvement of 0% [0%, 0%]. Consequently, no Phase-6D teacher was authorized, T6.26.1 and T6.26.2 were Dropped, and no learned training, checkpoint, quantization or formal-retention result was created. The final GO_RTL_ONLY verdict therefore does not depend on a CNN or student.

We have not hidden the favorable historical learning results. In a different finite-cutoff, two-level sBs controller task, a 72,853-parameter GRU teacher was distilled to a four-state, 95-scalar recurrence with evaluation action-imitation MSE 6.083136e-6 and a minimum matched retention point/lower bound of 0.981457/0.944501. Those results have a different observation, objective, simulator and action signature from the current multimode posterior/MLD task. They are therefore reported only as task-local historical evidence and cannot validate the current algorithm or the exact RTL. The full quantized-GRU route was also dropped after a 72,854-cycle optimistic lower bound without functional RTL or physical-retention evidence.

We also reran the preserved tiny CNN bit-exact five times on its 206-sample held-out residual-parameter split. It reduced parameter MSE from 8.034045e-6 to 2.414453e-6, but this was a single legacy split without an independent seed-cluster confidence interval and measured neither LER nor control gain. An exhaustive 16-family eligibility registry found zero same-task eligible learned checkpoints. We therefore retain the replay as a diagnostic method detail, not as evidence against simulator overfitting in the current task.

A future learned module can be reconsidered only as a replaceable approximation to an already authorized posterior, log-likelihood ratio, logical-coset probability or action. It must share the registered split and observed-only information contract, beat a matched classical approximation under the same runtime and memory budget, and pass calibration, action-agreement, LER-retention, worst-family, held-out-OOD, quantization and formal-retention gates while providing a concrete compression or cost benefit. Otherwise it remains an ablation. Thus, the present claims avoid the simulator-overfitting concern by being independent of learning; they do not claim that every future learned model will generalize.

## Evidence audit

- Current verdict: `GO_RTL_ONLY`; learning=`DROPPED_ABLATION_ONLY` and changes-overall-verdict=`false`.
- Current multimode entry: `NO_GO_MULTIMODE_CAUSAL_HEADROOM`, strongest=`static_mixture_exact_mld`, baseline/proposed=0.1119791667/0.1119791667, point/LCB=0.0%/0.0%.
- Legacy teacher: 72,853 parameters, `GRU10-DENSE256-DENSE256-OUT15`, cap hits=[0, 2].
- Legacy CNN: 206 samples, five bit-exact repeats, active/zero MSE=2.41445285e-06/8.0340452e-06; same-task eligible=0/16.
- Legacy student: 4 states / 95 scalars, evaluation MSE=6.08313616e-06.
- Legacy retention: minimum point/CI-lower=0.981457/0.944501; mismatch minimum=0.897630.
- Hardware boundary: full quantized GRU lower bound=72,854 cycles and ineligible; historical selected student=distilled_student_q3_14_state4_serial, but present-in-current-RTL=false.

## Manuscript change checklist

- Keep the two primary lanes explicit in Abstract and Introduction.
- Keep the Phase-6D teacher/training/formal absence explicit in Methods and Results.
- Keep legacy positive results task-local and non-migrating in the response and Supplement.
- Keep future promotion fields and ablation fallback explicit in Discussion.

## Missing information / risk flags

- `ACTUAL_REVIEWER_ID_AND_VERBATIM_WORDING`: scientific substance is complete, but the package must not be labelled submission-ready until the actual comment is supplied.
- The response establishes deletion-invariance of current claims with respect to learning; it does not establish universal generalization of any future CNN/student.

## 中文核对

- 当前论文不是 CNN-centric：两个主 lane 分别由 multimode strongest-baseline gate 和 exact single-mode RTL gate 决定，learning 没有投票权。
- 不能隐藏 T4.4 的正结果；必须明确它们属于另一 finite-model controller task，不能迁移成 Phase 6D multimode/RTL 证据。
- 实际返修信提交前，只需替换 reviewer ID 并粘贴原始评论；不要虚构编辑决定、reviewer 身份或行号。

## 原子证据与边界

| ID | 状态 | 主题 | 主张 | 边界 |
| --- | --- | --- | --- | --- |
| CNN-R01 | `REVIEWER_CONCERN` | comment_origin | The question is a pre-emptive reviewer risk, not a supplied verbatim report. | Keep the actual reviewer ID and verbatim wording as a visible placeholder. |
| CNN-R02 | `CURRENT_PRIMARY` | paper_center | The current paper has two primary lanes: multimode software qualification and exact single-mode deterministic RTL. | Learning is not a third primary lane. |
| CNN-R03 | `CURRENT_NEGATIVE` | multimode_entry_stop | The Phase-6D multimode entry test had zero causal headroom over static-mixture exact MLD. | The result did not authorize a downstream teacher or formal split. |
| CNN-R04 | `CURRENT_NEGATIVE` | learning_disposition | T6.26.1 and T6.26.2 are Dropped and the Phase-6D learned extension is absent. | No Phase-6D training, checkpoint, quantization or formal retention result exists. |
| CNN-R05 | `CURRENT_PRIMARY` | verdict_independence | The learned extension has no vote and cannot change GO_RTL_ONLY. | Removing every learned artifact leaves both primary lane decisions unchanged. |
| CNN-R06 | `LEGACY_POSITIVE` | legacy_teacher | A 72,853-parameter bounded-residual GRU teacher passed its historical finite-model task. | It is a finite-cutoff two-level sBs controller, not the current multimode decoder teacher. |
| CNN-R07 | `LEGACY_POSITIVE` | legacy_student | A four-state, 95-scalar student achieved evaluation action-imitation MSE 6.083136e-6. | Imitation accuracy is not current multimode LER or exact-RTL evidence. |
| CNN-R08 | `LEGACY_POSITIVE` | legacy_retention | Historical matched finite-model retention had minimum point 0.981457 and lower bound 0.944501. | The result is task-local and cannot be migrated across simulator, objective or action signature. |
| CNN-R09 | `RISK_DISCLOSURE` | legacy_ood | Randomized mismatch retained a minimum relative retention of 0.897630 on its qualifying cells. | Relative retention coexists with absolute degradation and does not establish universal OOD robustness. |
| CNN-R10 | `DEPLOYMENT_BOUNDARY` | full_gru_hardware | The quantized GRU route was dropped after a 72,854-cycle optimistic lower bound and no functional RTL or physical retention. | A post-route workload lower bound is not a deployed learned controller. |
| CNN-R11 | `LEGACY_POSITIVE` | legacy_student_hardware | The historical four-state student was the only eligible learned hardware candidate in its own task. | It is not present in the current exact production RTL and has no Phase-6D vote. |
| CNN-R12 | `TASK_SIGNATURE_BOUNDARY` | nonmigration | Legacy controller action imitation and current multimode posterior/MLD approximation are different tasks. | Positive legacy evidence is disclosed but cannot be promoted by analogy. |
| CNN-R13 | `PROMOTION_GATE` | authorized_targets | A future learned module may approximate only an authorized posterior, LLR, coset probability or action. | Truth, scenario identity and future labels are forbidden online inputs. |
| CNN-R14 | `PROMOTION_GATE` | data_split | Teacher and approximation must share the newly registered train/calibration/pilot/formal split. | Legacy checkpoints and Phase-6D evaluation data cannot select the model. |
| CNN-R15 | `PROMOTION_GATE` | matched_budget | Promotion requires a matched classical approximation budget and a real compression or cost benefit. | Parameter compression alone cannot rescue failed LER or safety gates. |
| CNN-R16 | `PROMOTION_GATE` | retention_metrics | Promotion requires calibration, action agreement, LER retention, worst-family and held-out OOD retention. | Imitation MSE alone is insufficient. |
| CNN-R17 | `PROMOTION_GATE` | implementation_metrics | Runtime, memory, quantization error and formal retention lower bound must be reported. | Learned accuracy does not prove atomic or fail-closed RTL behavior. |
| CNN-R18 | `MANUSCRIPT_CHANGE` | main_text | Abstract, Introduction, Methods and Discussion explicitly remove CNN from the paper center. | The answer maps to named sections, not invented line numbers. |
| CNN-R19 | `MANUSCRIPT_CHANGE` | supplement | Supplement tables label CNN/student as Dropped ablation with no vote in either lane. | A status row is not a performance claim. |
| CNN-R20 | `RESPONSE_WORDING` | direct_answer | The project is contract-centric; the overfitting risk is avoided for current claims by making them independent of learning. | This does not establish that every future learned model generalizes. |
| CNN-R21 | `RISK_DISCLOSURE` | unresolved_future | A future learned approximation still requires new held-out and OOD evidence. | Until that gate passes, it remains ablation or future work. |
| CNN-R22 | `RESPONSE_WORDING` | submission_readiness | The scientific substance is evidence-complete as a pre-emptive answer. | Submission readiness remains draft_with_placeholders until an actual reviewer ID and verbatim wording exist. |
| CNN-R23 | `LEGACY_POSITIVE` | legacy_cnn_replay | The preserved tiny CNN was rerun bit-exact five times on 206 held-out residual-parameter samples and improved parameter MSE over zero residual. | The single legacy split has no independent seed-cluster CI and measures parameter regression, not LER or control gain. |
| CNN-R24 | `TASK_SIGNATURE_BOUNDARY` | legacy_cnn_eligibility | The exhaustive learned/controller registry found zero same-task eligible checkpoints among 16 candidate families. | The exact legacy replay remains diagnostic and cannot enter the current learned-decoder ranking. |
