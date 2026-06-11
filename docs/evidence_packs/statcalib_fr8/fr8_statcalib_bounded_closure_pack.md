# FR8 StatCalib Bounded Closure Pack

## Verdict

`T70` 完成了一个只读、代码驱动的 `FR8` 收口包。它没有新建 run root，没有重跑 benchmark，也没有改写任何历史 `runs/` 产物；它只是把 `T24/T64/T66/T67/T68/T69` 的已接受 artifact 与 review 链收拢成一个可复算的 closure/gate 结果。

当前最强且仍然诚实的主线结论是：

1. `T24` 继续是 authoritative frozen ranked table。
2. `statcalib` 继续只是 separately labeled extension lane。
3. `T69` 之后最强 clean answer 仍然是 `statcalib_window_variance_t001 = statcalib_window_variance_t003 = statcalib_window_variance_t005`。
4. 没有 unique clean reference point。
5. 当前 promotion gate 结论是 `no_promotion_keep_extension_lane_only`。
6. 当前 unique-threshold gate 结论是 `future_selection_task_required`。

## Closure Table

| Category | Subject | Summary |
| --- | --- | --- |
| `frozen_anchor_evidence` | `T24` | 冻结主表继续有效；四个冻结场景 winner 都是 `hybrid_residual_b`，runner-up 都是 `ukf`。 |
| `extension_lane_evidence` | `T64/T66/T67/T68/T69` | extension lane 的 bounded win 已存在、对局部灵敏度不脆弱、对 `teacher_mode=ukf` 不强依赖、存在 full generated-only winners，并最终收口到 persistent clean tie set。 |
| `supported_claims` | 当前 `FR8` 有界答案 | `T69` 后的最强 clean answer 仍是 `window_variance_t001 = t003 = t005`，且没有 unique clean reference point。 |
| `unsupported_claims` | 越界转述 | 当前证据不支持改写 `T24`、不支持宣布唯一阈值、也不支持把 `statcalib` 升格成 mature comparator / `.tflite` / real-board / paper-grade expanded benchmark 证据。 |

## Frozen Anchor Evidence

helper 复算了 `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743` 的 `summary.json + comparison.csv`，并确认：

- `missing_runs = []`
- 20 个冻结 comparison rows 全部存在
- 四个冻结场景 winner 都是 `hybrid_residual_b`
- 四个冻结场景 runner-up 都是 `ukf`
- `T24` 仍然只代表 mock-backed software-HIL formal software revalidation，不是 `.tflite`，不是真板

因此，`T70` 不做也不允许做的第一件事，就是把 `T64/T66/T67/T68/T69` 反写成新的 frozen ranked table。

## Extension-Lane Evidence Chain

### T64

- `T64` 证明了在锁定四场景协议下，`statcalib` 作为第六 lane 可以在不改写 `T24` 冻结五模式子表的前提下单独报告。
- helper 重新比对了 `T64` 与 `T24` 的冻结 20 行，结果仍然是 exact match。
- 同时，`T64` 中 `statcalib` 在四个锁定场景里都优于 `ukf` 和 `hybrid_residual_b`。

### T65

- `T65` 不是新 benchmark，而是把 `T64` 的 report/artifact consistency 收紧成代码审计 guard。
- `T70` 把 `T65` 作为证据链的一部分，是因为后续复用 `T64` 时不能再靠人工 prose 漂移。

### T66

- `T66` 证明 `T64` 的 bounded advantage 不是一个单点参数偶然值。
- helper 复算到：
  - `best_by_mean_ler = statcalib_high_threshold`
  - `best_by_stability = statcalib_default`
  - 两者都不改变核心事实：bounded statcalib advantage persists

### T67

- `T67` 证明该 bounded advantage 不是 grossly tied to `teacher_mode=ukf`。
- helper 复算到：
  - `any_non_ukf_variant_beats_both_frozen_anchors_all_scenarios = true`
  - 两个 parameter point 下的最优 teacher anchor 都是 `window_variance`
- 这一步缩窄了“也许只是 `ukf` teacher 偶然带来的结果”这一类解释。

### T68

- `T68` 证明预声明网格里确实存在 full generated-only winners。
- helper 复算到的 full generated-only winner set 是：
  - `statcalib_window_variance_t001`
  - `statcalib_window_variance_t003`
  - `statcalib_window_variance_t005`
  - `statcalib_ekf_t001`
- 但 `T68` 还保留一个关键信息：`mean-best` 与 `worst-case-best` 当时并不相同。

### T69

- `T69` 关闭了 bounded clean-winner tie-break execution question。
- helper 复算到：
  - `final_clean_winner_classification = persistent_clean_tie_set`
  - `t68_clean_tie_set_relation = persists`
  - `mean_best_and_worst_case_best_relation = same`
  - `unique_clean_reference_point_exists = false`
- 所以 `T69` 的价值不是“终于找到了唯一阈值”，而是把 persistent tie 的 bounded 可信度做得更强。

## Supported Claims

当前 `FR8` closure pack 只支持以下说法：

1. `T24` 仍然是 authoritative frozen ranked table。
2. `T64` 提供了一次 provenance-clean 的 bounded extension-lane win。
3. `T66` 表明该 bounded win 在预声明局部灵敏度网格内仍能保持。
4. `T67` 表明该 bounded win 不粗略依赖 `teacher_mode=ukf`。
5. `T68` 表明预声明网格里存在 full generated-only winners。
6. `T69` 表明最终最强 clean answer 是 persistent tie，而不是 unique threshold。

## Unsupported Claims

当前 `FR8` closure pack 明确不支持以下说法：

1. 用 `T64/T66/T67/T68/T69` 改写 `T24` 历史 frozen main table。
2. 把当前 `statcalib` lane 升格为 mature calibration comparator。
3. 宣布已经得到唯一最终阈值。
4. 把当前证据外推成 `.tflite`、real-board 或 paper-grade expanded benchmark 结论。

## No-Promotion Gate

### Verdict

`no_promotion_keep_extension_lane_only`

### Why

1. `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` 仍明确要求：`statcalib` 不能默认加入 frozen `T24` ranked set。
2. `T64/T66/T67/T68/T69` 的所有 accepted 结果都仍是 mock-backed software-HIL extension-lane evidence only。
3. `T68/T69` 后 broader predeclared grid 依然不是 uniformly clean closure story。
4. 当前 review 链没有任何一项把 `statcalib` 从 extension lane 改判成 mature comparator。

所以，`T70` 的 promotion gate 不是“差一点就 promotion”，而是当前主线必须显式保持 `no_promotion`。

## Unique-Threshold Gate

### Verdict

`future_selection_task_required`

### Why

1. `T69` 的最终分类就是 `persistent_clean_tie_set`。
2. `T69` 明确给出 `unique_clean_reference_point_exists = false`。
3. 现在若硬选一个阈值，只能依赖外加准则，而不能再说“这是 T69 自己已经给出的唯一答案”。

因此，当前唯一诚实的 gate 结论不是“支持 unique threshold”，而是“若未来真要选一个，必须另开 selection-criterion task”。

## What A Future Task Would Need

如果后续真的要从 `t001/t003/t005` 里强行选一个单一阈值，最小前提至少包括：

1. 先预声明 selection criterion，而且这个准则不能还是当前已经并列的 `T69` mean/worst-case LER。
2. 先锁定候选集和决策规则，再执行或再复用证据，不能事后从 `T69` prose 倒推。
3. 在后续 promotion gate 明确批准之前，`T24` 仍保持 frozen，`statcalib` 仍保持 extension lane。
4. 如果目标 claim 想超过 mock-backed software-HIL 边界，就必须为那个新边界单开任务，而不是复用 `T64-T69` 直接升格。

## Read-Only Integrity

本次 `T70` helper 单次执行只读取以下主输入：

- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_consistency_audit.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_sensitivity_bounded_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_teacher_anchor_bounded_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_generated_only_robustness_bounded_benchmark.md`
- `docs/evidence_packs/statcalib_fr8/statcalib_clean_winner_tiebreak_bounded_benchmark.md`
- `docs/review/T64_review.md` 到 `docs/review/T69_review.md`
- `runs/p4_benchmark/T24...`
- `runs/p4_benchmark/T64...`
- `runs/p4_benchmark/T66.../statcalib_sensitivity_summary/summary.json`
- `runs/p4_benchmark/T67.../statcalib_teacher_anchor_summary/summary.json`
- `runs/p4_benchmark/T68.../statcalib_generated_only_summary/summary.json`
- `runs/p4_benchmark/T69.../statcalib_clean_winner_tiebreak_summary/summary.json`

并且 helper 输出里显式记录：

- `no_new_run_root_created = true`
- `historical_runs_modified = false`
- `sidecar_outputs_used = false`

这就是 `T70` 的收口边界：只读 consolidation，不新增执行事实。
