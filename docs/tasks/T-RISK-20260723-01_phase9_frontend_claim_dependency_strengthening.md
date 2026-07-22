# T-RISK-20260723-01：Phase 9 frontend、claim 与依赖链补强

- **Task ID：** T-RISK-20260723-01
- **标题：** 补齐 raw-IQ 数字前端、scoped SOTA 状态和 Phase 9 可执行依赖
- **日期：** 2026-07-23
- **状态：** Done
- **来源风险：** R-N170—R-N175

## 输入材料

- 用户要求：即使暂时得不到 Puviani checkpoint、20-agent seeds、selection ledger 和六态 evaluator，也继续执行双后端数字孪生、trusted codebook、同预算模型 tournament、observed-only posterior、六周期原子集成、六态长序列和高速板 HIL；
- `docs/new_task_board.md` 的 Phase 9 / T9.1—T9.8；
- `docs/experiment_plan.md §20.1--§20.7`；
- 已封存的 `docs/t9_1_1_three_lane_protocol.json`，其中 `analysis_sha256=c88110375c358794339e72d672e4624871425fe480e5da091ddd1d6595255e18`；
- 当前运行中的 T9.1.3 paper-constrained production lineage；
- 既有 T6.25 single-mode RTL：输入边界是离散 control word/discriminator 后接口，并不含 raw-IQ matched-filter/discriminator；
- 两轮只读深审：一轮核对七步路线，一轮核对 T9.1.1 semantic projection、治理测试和当前训练 hash 级联。

## 执行前方案

1. 不修改 T9.1.1 的 frozen config、生成器、report、Source Data 或 `analysis_sha256`，确保当前 T9.1.3 checkpoint/lineage 继续有效。
2. 将新的 claim taxonomy 放入未来 `T9.1.5` parent-bound child amendment；旧 `GO_LER_SOTA/GO_LIFETIME/GO_HIL_SPEED` 仅保留为 v1 内部 candidate gates。
3. 在 codebook 之前新增 platform/raw-IQ interface freeze 和可综合 frontend 资格，明确 physics IQ source 不等于 deployable discriminator。
4. 修复 T9.2.4 在 codebook 尚未生成时引用 `codebook actions` 的顺序环：先用 conservative representative probes，对最终 codebook 的资格留给 T9.3.4。
5. 将六周期严格限制在 `discriminator-out -> action`；ADC/raw-IQ、matched filter、discriminator、CDC/transport 和 trigger 分别计时。
6. 让 T9.8 接受真实板 lane 的 terminal `Blocked/null`，保证无板不阻塞 algorithm-only GO/NO-GO；同时让 measured-speed 保持 null。
7. 删除 T9.7.3 对旧 T6.9.2 Route-A/GW2AR 板测的硬依赖，只允许复用其 transport/fault 经验。
8. 增加集合级治理断言，防止以后悄悄加 task 而 34-task 硬编码测试仍误通过。

## 实际完成内容

### 1. 新增三个正常任务

- `T9.1.5`：以 T9.1.1 immutable hash 为 parent，建立 scoped claim child seal。显式区分：
  - `GO_LER_REGISTERED_BEST` 与 `GO_LER_EXTERNAL_SOTA`；
  - `GO_LIFETIME_PROJECT_NATIVE`、`GO_LIFETIME_EXTERNAL_SOTA` 与 `GO_PHYSICAL_LIFETIME`；
  - `GO_HIL_INTEGRATED` 与 `GO_HIL_EXTERNAL_SPEED`。
- `T9.2.6`：在 codebook 前冻结 raw-IQ frontend、stream/CDC、Q-format、四种 latency boundary 和候选平台共同 envelope。
- `T9.2.7`：实现 independent Python/NumPy golden 与 synthesizable matched-filter/discriminator RTL，要求 held-out ROC/AUC、NLL/Brier/ECE、confusion/abstain、float-to-fixed/domain-gap 报告、CXXRTL 逐 bit 对拍、故障注入和 targeted mutation；没有真实 recorded/live IQ 时对应证据保持 null。

Phase 9 从历史初版 34 个 task 增加到 37 个；`T9.1.2/T9.7.3/T9.7.4` 仍是仅有的局部 Blocked task，当前推荐任务仍是 T9.1.3。

### 2. 修复现有任务的顺序与证据边界

- T9.1.4 增加检索截止日、检索式、去重/排除 ledger、same-task eligibility 和不可运行 baseline 的 null 原因。
- T9.2.1 先冻结 conservative representative action probes；T9.2.4 只用这些 probes 做 twin intervention，不提前声称 codebook coverage。
- T9.3.1/T9.3.3 强制消费 T9.2.6—T9.2.7 的平台与前端 envelope。
- T9.4.1 的 deployable IQ/LLR 必须来自同一 bit-accurate frontend；continuous/ideal simulator IQ 只作 privileged ceiling，synthetic/recorded/live-raw 分栏。
- T9.6.5 先输出 registered/project-native promotion；external/Puviani/physical 状态需要额外资格。
- T9.7.1 从 raw/recorded IQ 开始 integrated CXXRTL，并分开报告 frontend 与 6-cycle fast-path boundary。
- T9.7.2 只能选择满足早期 envelope 的平台；不兼容时必须 NO-GO/显式 amendment。
- T9.7.3 移除旧 T6.9.2 hard dependency。
- T9.8.1 将 T9.7.4 解释为 `Done` 或 terminal `Blocked/null` evidence input；T9.8.2 允许无板时给 algorithm-only verdict，但禁止 measured-HIL/单篇硬件 claim。
- T9.8.3 的交付清单增加 raw-IQ golden/frontend RTL 与相关 manifests。

### 3. 低频计划、风险和入口同步

- `docs/experiment_plan.md` 新增 §20.8；不修改被 T9.1.1 semantic binding 捕获的 §20.1/20.2/20.6/20.7。
- `docs/new_risks.md` 新增 R-N170—R-N175：
  - R-N170/R-N171/R-N174 保持 Open，等待实际 frontend、child claim seal 和 platform envelope；
  - R-N172/R-N173/R-N175 因依赖语义已修复记为 Mitigated，但执行时仍需机器验证。
- `README.md` 将计划范围更新到 §20.8，并加入本补强记录的文档地图。
- `tests/test_new_task_board_governance.py` 将 Phase 9 预期集合从 34 更新为 37，并新增 actual-set equality、probe/codebook、旧板依赖、可空 HIL 和 scoped claim/risk 断言。

## 产物路径

- `docs/new_task_board.md`
- `docs/experiment_plan.md`
- `docs/new_risks.md`
- `README.md`
- `tests/test_new_task_board_governance.py`
- `docs/new_tasks/T-RISK-20260723-01_phase9_frontend_claim_dependency_strengthening.md`
- `docs/tasks/T-RISK-20260723-01_phase9_frontend_claim_dependency_strengthening.md`

## 验证方式和结果

- Phase 9 定向治理测试：
  - `pytest tests/test_new_task_board_governance.py::test_phase9_performance_first_single_mode_reboot_is_nonblocking_and_fail_closed`
  - 结果：`1 passed`。
- 完整治理 + frozen protocol 回归：
  - `pytest tests/test_new_task_board_governance.py tests/test_phase9_three_lane_protocol.py`
  - 结果：`34 passed`。
- T9.1.1 repository live verifier：
  - `python -m cnn_fpga.benchmark.phase9_three_lane_protocol --verify`
  - 结果：identity/all_gates/gate_cache/verdict/analysis_hash/source_data/markdown_live/current_results_null 八项全部 `true`；v1 `analysis_sha256` 未改变。
- Phase 9 机械审计：`37 rows / 37 unique / 0 duplicates`；状态为 `1 Done / 1 In Progress / 32 Todo / 3 Blocked`。
- 双完成记录逐 byte 相同；`git diff --check` 无 whitespace error，仅报告工作区既有 LF/CRLF 转换提示。
- 定向语义断言确认：
  - T9.7.3 source 不再包含 T6.9.2，正文只允许把它作为旧 transport 参考；
  - T9.8.1 接受 T9.7.4 terminal `Blocked/null`；
  - T9.2.4 不再引用最终 `codebook actions`；
  - actual Phase 9 task ID 集合必须严格等于预期 37 项，不能再靠遗漏硬编码静默通过。
- 回归首次故意发现并拒绝了 T9.6.5 source 列的增量漂移（G29 fail）；随后把 R-N171 约束保留在 acceptance/§20.8、恢复 frozen source projection。该失败证明 parent seal 确实在工作，而不是只比较静态字符串。

## 非简化实现复核

本次补强专门阻止四种 demo 化路径：

1. 不能用 simulator truth 或预构造 label 直接产生 discriminator output；
2. 不能只实现 normal-case passthrough frontend；必须覆盖量化边界、overflow、sample gap/reorder、CDC/backpressure、版本/CRC、reset 和可杀 mutation；
3. 不能把“超过本项目实现的若干 baseline”缩写为 external SOTA；必须有可审计检索和完整 same-task eligibility；
4. 不能把 6-cycle core 数值复制到 raw-IQ-to-trigger，也不能用旧板或 CXXRTL 填 measured 字段。

由于本 task 是计划/治理补强，而不是 frontend 实现本身，R-N170/R-N174 保持 Open；真正的反 demo 结论要等 T9.2.6/T9.2.7 交付 code、raw vectors、mutations、CXXRTL 与综合证据后才能关闭。

## 风险复核

- R-N170：Open/Critical/Immediate；raw-IQ frontend 断链，由 T9.2.6—T9.2.7/T9.7.1 关闭。
- R-N171：Open/Critical/Immediate；registered-best 外溢为 external SOTA，由 T9.1.4—T9.1.5/T9.8 关闭。
- R-N172：Mitigated/High/Soon；无板终门伪硬依赖已修正，待 child seal/mutation 验证。
- R-N173：Mitigated/High/Soon；codebook 顺序环已拆成 primitive probes 与最终 coverage。
- R-N174：Open/Critical/Immediate；平台约束冻结过晚，由 T9.2.6 和下游 envelope binding 关闭。
- R-N175：Mitigated/High/Soon；旧 T6.9.2 hard dependency 已删除。

## 是否需要继续插入 task

不需要。三个新增正常 task 和对既有 downstream gate 的修正已经覆盖六项风险。后续按 T9.1.3 → T9.1.4/T9.1.5 → T9.2 主链执行；不得新增替代 official assets、跳过 frontend、跨 latency boundary 或伪 measured 的旁路。

## 对任务板的同步

- 插入任务区新增 T-RISK-20260723-01 并标记 Done；
- Phase 9 增加 T9.1.5、T9.2.6、T9.2.7，总数 37；
- 当前推荐任务与 T9.1.3 In Progress 状态不变；
- T9.1.1 immutable parent、Puviani/external/physical null 边界和三个局部 Blocked 状态不变；
- 进度日志新增本次审计开始与完成记录。
