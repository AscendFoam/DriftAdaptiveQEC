# Risks And Open Questions

## 风险清单

| ID | Risk | Level | Evidence | Mitigation |
| --- | --- | --- | --- | --- |
| R1 | 默认运行环境不可直接执行最小 benchmark | 中 | 默认 `python 3.13.7` 仍缺 `numpy`，但 `C:\ProgramData\anaconda3\python.exe` 已可跑通 P0 smoke | 后续所有治理文档继续显式指定推荐解释器 |
| R2 | 根目录虽已补 recovery-scoped manifest，但完整训练链、`.tflite` 与真板环境仍无统一依赖说明 | 中 | `requirements-recovery.txt` 只覆盖 `P0/P3/P4 recovery smoke`，且显式不含 `torch`、`tensorflow`、`tflite-runtime`；`docs/training_chain_bootstrap.md` 已补训练链 bootstrap，但还不是跨机器完整依赖锁定 | 继续保持作用域诚实；训练链已独立说明，`.tflite` 与真板路径仍需单开有界 manifest / bootstrap 任务 |
| R3 | 软件 HIL 与真板 HIL 边界容易被误写 | 高 | `cnn_fpga/hwio/board_backend.py` 仍是 placeholder 风格；`docs/03_hil_p4_boundary_audit.md` 已完成边界澄清；`T20` 当前任务只允许补 readiness checklist，不允许实现或宣称真板完成 | 后续所有文档、复验与报告都必须引用 `docs/03_hil_p4_boundary_audit.md` 的统一口径；`T20` 只做只读 readiness 审计 |
| R4 | 仓库中仍有历史生成物噪声，容易混淆当前事实来源 | 中 | `.gitignore` 已忽略 `__pycache__`、`runs/`、`artifacts/`；`T33` 已将先前 `116` 个 tracked `.pyc` 文件从 Git index 中移除，因此 tracked cache/bytecode 已归零，但 Git 中仍有 `1841` 个已跟踪 `runs/` 文件、`110` 个已跟踪 `artifacts/` 文件 | 对 tracked cache 的 bounded cleanup 已完成；后续如需处理 `runs/` / `artifacts/`，必须单开独立 cleanup 任务，并继续禁止把整目录改写成新的事实来源 |
| R5 | P4 目前已完成四场景、五模式、`repeats=2` 的 formal software revalidation，但仍为 mock-backed software HIL，不是 `.tflite` runtime、不是 `real_board`、不是 paper-grade expanded benchmark | 中 | T24 run dir: `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`；`missing_runs = []`，`coverage = 1.0`；四场景 winner 均为 `hybrid_residual_b` | T24 已完成 formal frozen-set software revalidation；但仍不可写成 paper-grade benchmark 或 runtime/board validation；后续如要升级证据等级，需新增 statcalib comparator、CI-driven stopping 或 runtime/board 路径 |
| R6 | `.tflite` 真导出与 stub 回退容易混淆 | 中高 | `cnn_fpga/model/export.py` 与 `cnn_fpga/runtime/inference_service.py` 同时支持两种路径，且 runtime 输出不同 `source`；`T4/T7` 当前都刻意未走 `.tflite` 路径 | 文档与日志必须显式标注 `artifact type`，并区分 `tflite_service` 与 `tflite_stub_service` |
| R7 | `T19` manifest 对应的 tracked-cache physical cleanup execution | 已收口 | `T33` 已按 `docs/cleanup_tracked_cache_manifest.md` 对 9 个 manifest-listed `__pycache__` 目录执行 bounded untrack；`git ls-files | rg "__pycache__|\\.pyc$"` 已归零 | `R7` 对 tracked-cache lane 已关闭；`runs/` / `artifacts/` 如需处理仍必须另开任务 |
| R8 | 最小 software HIL 路径虽然已在 bounded recovery path 上完成逐字一致复验，但该结论容易被误外推到真板、`.tflite` 或正式 benchmark | 中 | `T12` 已确认 `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104` 与 `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104` 的 `hil_summary.json` / `hil_events.json` 哈希一致；但路径仍固定为 `mock + model_artifact + artifact_npz + inproc` | 后续文档必须继续写清结论边界，不把 bounded recovery smoke 扩写成真板或正式 benchmark 已恢复 |
| R9 | T24 已完成 formal frozen-set revalidation，但若继续扩大到更多 repeat、CI-driven stopping 或 extra drift families，仍可能隐式越过 frozen-set/formal 边界 | 中 | T24 已按 locked protocol 跑完 `4 scenarios x 5 modes x repeats=2`；`docs/P4_benchmark_formal_protocol.md` 已锁定边界 | T24 后仍不应自动追加更大 repeat、额外 scenario 或 statcalib comparator；任何进一步 P4 扩展都必须新开任务包 |
| R10 | `hybrid_residual_b` 的机制证据虽然已经补到 multi-seed diagnostic generalization 与一次 bounded targeted intervention evidence，但当前机制解释仍不能闭环，因为 `T55` 表明“high committed-`b` is harmful”并不是可泛化的一般解释，后续需要 claim reframing 与更清晰的 causal isolation | 中 | `docs/review/T27_teacher_diagnostics_path_audit.md` 已将主因缩窄为 broadcast teacher features 不触发 scalar explain diagnostics；`docs/review/T28_review.md` 确认当前输出已显式标记 `not_generated` / `not_applicable`；`docs/seed20260429_failure_diagnosis.md` 与 `docs/seed20260429_trace_export_diagnosis.md` 已把 `20260429` 缩窄为 trace-supported committed combined-`b` instability；`docs/multi_seed_trace_generalization_probe.md` 与 `docs/review/T54_review.md` 已确认该模式在 6-seed pack 中为 broadly repeated with qualifications；`docs/multi_seed_i1_intervention_probe.md` 与 `docs/review/T55_review.md` 已表明 pure I1 lower-clip intervention 为 mixed 且整体偏 harmful；`docs/review/T56_review.md` 已把机制叙事明确收口为 retain / weaken / retire / reframe / still-open | R10 remains open but changes character after `T56`：问题不再是缺 intervention evidence，而是已有 intervention evidence 使简单 harmful-instability 叙事站不住脚。`T56` 已完成 claim reframing；当前 `T47` 只能作为 hedge-conditioned paper-material lane 推进，不能写成机制闭环或第二个 intervention 自动开启 |
| R11 | 训练链 clean CPU-only environment 已完成 draft lock、dry-run/import bootstrap 与 one real-training smoke，但 full training reproducibility 与 broader portability 仍未验证 | 中 | `docs/training_chain_portable_dependency_lock_plan.md` 已由 T31 产出并通过 `docs/review/T31_review.md` `PASS`；`requirements-train-cpu-win-py312.txt` 与 `docs/training_chain_cpu_cleanenv_bootstrap.md` 已由 T39 产出并通过 `docs/review/T39_review.md` `PASS`；`docs/training_chain_cpu_cleanenv_train_smoke.md` 已由 T40 产出并通过 `docs/review/T40_review.md` `PASS`；clean env 与 `DLEnv` 分离，且已完成一次真实 CPU-only training smoke | 不把 T40 写成 full training reproducibility、GPU/CUDA portability、Linux portability 或 production-scale training validation proof；后续如要继续推进 training portability，必须单开新的 bounded task |
| R12 | `.tflite` 路径已有代码与入口，但真实 TensorFlow / TFLite 运行时在当前机器上不可用 | 高 | `docs/TFLite_runtime_bootstrap.md` 已记录 `tensorflow = False`、`tflite_runtime = False`；`export.py`、`evaluate_tflite.py`、`validate_export.py` 入口存在，但真实 runtime 需独立环境 | 继续把真实 `.tflite`、stub manifest 与 HIL benchmark 边界写清；若后续要跑真实 runtime，单开环境任务或在具备依赖的机器上做独立 smoke |
| R13 | 真板 HIL 入口存在配置骨架，但距离可执行真板 smoke 仍缺设备、权限、寄存器一致性与日志证据 | 高 | `board_backend.py` 仍是 placeholder；设备缺失时会触发 `board_device_missing:...`；`schedule_commit(...)` 仍返回 `target_bank=None`、`version=None`、`ack_delay_us=None`；`step(...)` 返回空事件；`docs/real_board_hil_readiness.md` 已固定前置条件与验收标准 | 后续若推进真板路径，必须单开执行任务，逐层补齐设备存在、寄存器活性、DMA 读出与 commit/ack round-trip 证据，在此之前禁止写成 real-board HIL 已完成 |
| R14 | T22 已把寄存器来源、DMA 审计清单和量化阈值草案具体化，但真实宿主、bitstream 与 DMA contract 仍未验证 | 中高 | `docs/real_board_smoke_execution_plan.md` 已直接映射 `axi_map.py` / `dma_client.py`，`docs/review/T22_review.md` 确认 AXI/DMA 审计清单与源码吻合；但 N2 指出 preflight 输出格式仍需改进，N3 指出 `byte_count = 4096` 依赖 `32 x 32 float32` 假设 | 后续若进入真板执行任务，必须先选择宿主模型，再用实际 bitstream / RTL / DMA contract 确认地址表、histogram shape、element dtype、timeout 与 commit/ack 阈值 |
| R15 | Phase 2 当前已完成一轮 milestone queue，但证据仍混合停留在 development / bootstrap / manifest / readiness 层，若直接升级到 formal benchmark、真实 `.tflite` runtime、physical cleanup 或 real-board validation，容易再次打破边界诚实 | 高 | `docs/review/T21_phase2_milestone_review.md` verdict = `Conditional`；`T15` 仍只是 `development_smoke`；`T18` 真实 runtime 不可用；`T19` 未执行 cleanup；`T20` 不是真板验证 | 保持 `Phase 2: Controlled Development` / `Go`，继续只开 bounded 下一任务；优先补 `T22` 这类 execution-plan 级文档任务，而不是直接进入高风险执行任务 |
| R16 | 把“最终要发论文”误压缩成最近任务直接写论文 claim，容易跳过 formal benchmark、机制诊断和部署边界证据 | 高 | `T24` 已完成 frozen-set formal software revalidation，`T25` 已确认其 result boundary，`T27/T28` 已缩窄并修复 teacher diagnostics 输出语义，`T29` 已修复人读 report header，`T26` 已完成 statcalib feasibility gate；但 `T18` 未恢复真实 `.tflite` runtime，`T22` 不是 hardware validation，R10 仍不是完整机制证据 | Paper claim/evidence ledger 仍应推迟到机制证据更清楚之后；当前 `T30` 只做 statcalib interface contract / bounded implementation package，不写论文 claim |
| R17 | 深度研究报告建议的 formal benchmark 范围可能显著扩大，若无分级采纳会把 benchmark expansion lane 变成不可执行的大任务 | 中高 | `docs/reference/进一步的深度研究结果.md` 建议加入强 classical / soft-information / calibration / learned baseline 类别、更多 drift families、训练/评测 seed 分离、置信区间、latency/commit/rollback 指标和 statcalib baseline；`docs/paper_benchmark_expansion_protocol.md` 已把它们分类为 adopted / deferred / rejected | T45 已完成 protocol lock，但未来若要真正扩 benchmark，仍必须单开 bounded execution task，并保持 frozen-set anchor 不变 |
| R18 | `T24` 已按 frozen-set scope 完成，但若后续把 `statcalib`、soft-information、额外 drift families、CI-driven stopping、`.tflite` runtime 或真板边界并入同一任务，仍会重新打破 scope | 中高 | `docs/P4_benchmark_formal_protocol.md` 已把 `T24` gate 锁为 `GO_FOR_BOUNDED_FORMAL_SOFTWARE_REVALIDATION` + `NO_GO_FOR_SCOPE_EXPANSION_INSIDE_T24`；T24 实际完成 matrix 为 `4 scenarios x 5 modes x 2 repeats`；T25 已接受该边界；T26 gate verdict = `CONDITIONAL_GO` for separate comparator lane only；T45 已明确 future expansion lane 仍需独立任务；T54 已完成 trace-only lane；T55 已完成 one-variant intervention lane；`docs/review/T56_review.md` 已确认后续机制解释必须留在边界内 | `T47` 只允许作为 docs-only 的 hedge-conditioned paper-material lane；不得把第二个 intervention、benchmark expansion、额外 comparator、`.tflite` runtime 或真板边界混成同一任务 |
| R19 | T24 formal execution 已固定 exact CLI 和报告了 metric availability | 已收口 | T24 已使用 repeat-chunked CLI shape，所有请求统计字段已存在于 `comparison.csv`；`correction_saturation_rate_mean` 全零、teacher diagnostics 全零已报告为缺口 | R19 已由 T24 Worker 收口；后续若 runner 更新指标路径，需重新验证 |
| R20 | `correction_saturation_rate_mean` 在 T24 所有 20 个 scenario/mode rows 中结构性为 0.0；T27 已证明它不共享 teacher diagnostics 死路径，但尚未证明所有参数区间都不会触发 | 中 | `docs/review/T27_teacher_diagnostics_path_audit.md` 指出该字段来自 `fast_loop_emulator.py` 独立 saturation counter，并由 HIL summary 转抄到 `comparison.csv`；当前 T24 更像现参数区间下 genuine zero | R20 remains open but materially narrowed；不在 T28 中扩大 stress run，后续如需证明触发性应单开 edge/stress 任务 |
| R21 | Teacher diagnostics downstream missing-vs-zero writer 语义已由 T28 修复 | 已收口 | `docs/review/T28_review.md` 确认 T28 smoke 中 `ukf` 为 `not_applicable`、`hybrid_residual_b` 为 `not_generated`，missing numeric teacher diagnostics 保持 empty/null，`correction_saturation_rate_mean = 0.0` 保持为独立 observed zero | R21 对当前 writer 语义关闭；未来若再次改 aggregation/report writer，应保留 `not_generated` / `not_applicable` / `true zero` 区分 |
| R22 | T28 后 `_write_report()` markdown report 存在重复 header row，导致人读 report 表格列数不一致 | 已收口 | `docs/review/T29_review.md` verdict = `PASS`；旧 11-column header 已删除；验证得到 `header_rows=1`、`column_counts=[12, 12, 12]` | R22 已由 T29 收口；未来若再改 aggregation/report writer，应按 R23 补 focused test 或静态 report-shape check |
| R23 | Aggregation/report writer 缺少 focused unit/static tests，未来可能再次出现格式或 null-semantics 回归 | 中 | `docs/review/T28_review.md` Missing Tests 指出相关路径没有现成 tests；T28 依赖 py_compile 和 bounded smoke 验证 | T28 可接受；后续再改 aggregation/report writer 时应补 focused unit test 或静态 report-shape check |
| R24 | Current `statcalib` lane is still only a bounded extension-lane comparator path; if T30 helper, T59/T60/T62 smoke evidence, T64 bounded extension-lane win, or T66 local sensitivity outputs are overstated as a full statcalib/calibration comparator, the repo would overclaim unvalidated algorithm capability | 中 | `docs/review/T30_review.md` N4; `docs/review/T59_review.md`; `docs/review/T60_review.md`; `docs/review/T62_review.md`; `docs/review/T64_review.md`; `docs/review/T65_review.md`; `docs/review/T66_review.md`. T64 proves one clean bounded extension lane under the locked protocol, T65 hardens report/artifact consistency, and T66 shows that the bounded win survives one local five-point sensitivity grid; however, aggregate-best and stability-best variants differ, and the `static_bias_theta / statcalib_high_threshold` scenario-best row still carries aggregate `statcalib_status = mixed` | Keep `statcalib` labeled as a separately labeled extension lane in all FR8 docs, figures, and gates; keep the aggregate-best vs stability-best split explicit; do not upgrade T64/T65/T66 outputs into full calibration comparator, `.tflite`, or real-board evidence without a new bounded validation task |
| R25 | 论文叙事与 recovery baseline 曾一度跑在证据材料前面，若继续在未冻结 truth 的情况下推进 prose，很容易再次把 draft 当成事实来源 | 高 | `T43` 已经产出 bounded Background / Related Work prose draft，但用户明确要求改入 `Research Reality Recovery Mode`，优先补 claim/evidence/material/figure/reproducibility baseline | 先完成 `T44` recovery baseline，再决定是否恢复任何 prose 扩写；恢复前不把 draft、skeleton 或 framing 当成证据升级 |
| R26 | `T59` 的 cross-mode `teacher_mode` fallback leakage | 已收口 | `docs/review/T60_review.md` 已确认 `slow_loop.statcalib.teacher_mode` 不再泄漏到非 `statcalib` mode；`tests/test_statcalib_runtime_smoke.py` 新增了 mode isolation 回归覆盖 | `R26` 已由 `T60` 收口；未来若再改 `SlowLoopRuntimeConfig.from_config()`，必须保持 `statcalib.teacher_mode` 仅在 `mode=statcalib` 生效 |
| R27 | `statcalib` lane 缺 provenance-clean fairness sanity evidence | 已收口 | `docs/review/T59_review.md`；`docs/review/T60_review.md`；`docs/review/T61_review.md`；`docs/review/T62_review.md`；`docs/statcalib_comparator_lane_smoke.md`；`docs/statcalib_fairness_sanity.md`；`docs/statcalib_provenance_isolated_fairness_rerun.md`；`runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740/summary.json`；`runs/p4_benchmark/T61_statcalib_fairness_sanity_20260527_015239/summary.json`；`runs/p4_benchmark/T62_statcalib_provenance_isolated_20260527_122943/summary.json`。T59 是 dirty-worktree smoke；T61 clean launch 与 final `summary.json git_commit` 漂移；T62 则在 clean `main` 上完成单次 uninterrupted rerun，launch/finish/summary commit identity 全一致，且无 duplicate `running` repeat key | `R27` 已由 `T62` 收口。后续如需推进 `FR8`，仍必须经过独立 gate 任务，且不得把当前 bounded mock-backed software-HIL evidence 直接写成 formal comparator ranking |
| R28 | T64 result-pack report/artifact consistency gap | 已收口 | `docs/review/T64_review.md`; `docs/review/T65_review.md`; `docs/fr8_statcalib_extension_lane_consistency_audit.md`; `cnn_fpga/benchmark/audit_fr8_extension_lane_consistency.py`; `tests/test_fr8_extension_lane_consistency.py`. T65 repaired the wording drift, added an explicit audit helper, added focused regression coverage, and produced a bounded audit document | Closed by `T65`. Future reuse of the frozen T64 pack should keep the audit helper, focused test, and consistency-audit doc together rather than relying on manual wording alone |

## 当前开放问题

Current T24-T29 status note:

- `T24` Worker 已完成 formal software revalidation execution：`missing_runs = []`，20/20 `coverage = 1.0`，40 repeat-runs。
- Run dir: `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
- 四场景 winner 均为 `hybrid_residual_b`，runner-up 均为 `ukf`。
- `docs/review/T24_review.md` verdict = `PASS_WITH_WARNINGS`；Captain 已接受该结论并标记 T24 完成。
- Warning 分类：N1 correction saturation structural zero = `deferred` / R20；N2 task-board environment note = `accepted`；N3 teacher diagnostics header-only = `deferred` / R10。
- T24 仍为 mock-backed software HIL，不是 `.tflite` runtime、不是 `real_board`。
- `T25` Captain 已接受 gate review 为 `PASS_WITH_WARNINGS`；结论是 T24 可视为 completed frozen-set formal software revalidation，但边界仍严格限定为 mock-backed software HIL only。
- `T27` Captain 已接受 path audit 为 `PASS_WITH_WARNINGS`；R10 已缩窄为 broadcast teacher layout 与 scalar explain diagnostics 前提不匹配，R20 已缩窄为独立 fast-loop saturation path。
- `T28` Captain 已接受 review 为 `PASS_WITH_WARNINGS`；R21 对当前 writer 语义已收口，R10 进一步缩窄但不关闭。
- `T29` Captain 已接受 review 为 `PASS`；R22 对 P4 markdown report duplicate header 已收口。
- `T26` Captain 已接受 review 为 `PASS`；gate verdict = `CONDITIONAL_GO`，statcalib 只能作为 separate comparator lane 后续推进。
- `T30` Captain 已接受 review 为 `PASS`；已完成 statcalib interface-only contract 和 focused tests，但不等于 slow-loop integration、formal benchmark evidence、`.tflite` runtime 或 real-board validation。
- `T36` Captain 已接受 review 为 `PASS`；已完成 `seed=20260429` failure-mechanism diagnosis，结论仍是 summary/final-snapshot-level hypothesis，不是 causal proof。
- `T38` Captain 已接受 review 为 `PASS`；single-seed trace evidence 支持 `seed=20260429` 的 combined committed-`b` instability，但不是 mitigation、multi-seed causal proof、formal benchmark、`.tflite` runtime 或 real-board validation。
- `T31` Captain 已接受 review 为 `PASS`；已产出 `docs/training_chain_portable_dependency_lock_plan.md`，但不是 clean-environment rebuild proof。
- `T46` Captain 已接受 review 为 `PASS`；其非阻塞评论全部按 `accepted` 处理，没有 `deferred` warning。
- `T54` Captain 已接受 review 为 `PASS`；其非阻塞评论全部按 `accepted` 处理，没有 `deferred` warning；当前 multi-seed 结论是 broadly repeated with qualifications，`C4` 保持 `partial`。
- `T55` Captain 已接受 review 为 `PASS`；其非阻塞评论全部按 `accepted` 处理，没有 `deferred` warning；当前 intervention 结论是 mixed 且整体偏 harmful（harms 4/6, helps 2/6），`C4` 仍保持 `partial`。
- 当前唯一任务：`T47: Paper ablation result-pack and material ledger`，任务包 `docs/tasks/Phase2/T47_paper_ablation_result_pack_and_material_ledger.md`，且仅限 hedge-conditioned docs-only 推进。
- R13 当前仍然有效：真板路径还缺设备存在、权限、寄存器活性、DMA 读出和 commit/ack round-trip 的真实证据。
- R14 当前仍然有效但已收窄：AXI/DMA 代码侧审计已具体化，真实宿主、bitstream 与 DMA contract 仍未验证。
- R19 已收口：T24 已固定 CLI shape 并报告 metric availability。

1. 当前项目在这台机器上实际可用的 Python 环境是哪一个？
   - 当前答案：
     - P0/P3/P4 recovery smoke: `C:\ProgramData\anaconda3\python.exe`
     - torch 训练候选: `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
2. 历史文档中引用的 `.venvs/tf311` 是否在本工作区外部，还是已经失效？
   - 当前已知：工作区内未找到该路径
3. `T4/T6/T7` 的最小 recovery 复验路径，默认应该先选哪条组合？
   - 当前答案：
     - software HIL: `hil.backend=mock` + `model_artifact` + `artifact_npz` + `inproc`
     - P4 benchmark 最小路径: `p4_multiscenario_recovery_smoke.yaml` + `static_bias_theta` + `static_linear/cnn_fpga` + `paired_seeds`
     - P4 frozen baseline smoke: `p4_multiscenario_recovery_smoke.yaml` + `static_bias_theta` + `static_linear/window_variance/ekf/cnn_fpga` + `paired_seeds`
4. `T9` 的 `single-scenario / four-mode / repeats=1` 证据，是否已经足以支撑项目从 `Repair` 进入 `Go`？
   - 当前答案：在 `T10` 时点是否；但结合 `T11 + T12 + T13` 后，答案是可以进入“受控 `Go`”
5. 最小 software HIL bounded recovery path 是否已经收口到更严格的确定性复现？
   - 当前答案：是。`T12` 已完成，且两次新 run 的 `hil_summary.json` / `hil_events.json` 已逐字一致
6. 训练与 recovery benchmark 当前分别依赖哪些最小包集？
   - 当前答案：
     - recovery smoke root manifest: `numpy + PyYAML`
     - 训练链当前单独记录在 `docs/training_chain_bootstrap.md`，推荐解释器为本机 `DLEnv`
     - `.tflite` 路径当前单独记录在 `docs/TFLite_runtime_bootstrap.md`，真实 runtime 依赖尚未满足
7. 是否需要再为训练链、`.tflite` 或真板路径补独立 manifest？
   - 当前答案：训练链 bootstrap 已补；`.tflite` bootstrap 已补；真板路径仍需要后续独立任务
8. 已跟踪的 `.pyc` / `__pycache__/`、`runs/`、`artifacts/` 何时启动有界 cleanup，并如何拆分“bootstrap 必需”与“历史归档”？
   - 当前答案：
     - `T19` 已产出 `docs/cleanup_tracked_cache_manifest.md`，确认 tracked `.pyc` 共 `116` 个，全部位于 `9` 个 `__pycache__` 目录。
     - `T19` review verdict = `PASS`，但只制定 tracked cache cleanup manifest，不执行删除，不处理 `runs/` / `artifacts/` 物理清理。
9. 下一张继续开发任务包应该优先选哪一类？
   - 当前答案：
     - `T29` 已完成并由 Captain 接受为 `PASS`。
     - `T26` 已完成并由 Captain 接受为 `PASS`。
     - `T30` 已完成并由 Captain 接受为 `PASS`。
     - `T36` 已完成并由 Captain 接受为 `PASS`。
     - `T38` 已完成并由 Captain 接受为 `PASS`。
     - `T31` 已完成并由 Captain 接受为 `PASS`。
     - 当前唯一任务为 `T47: Paper ablation result-pack and material ledger`，任务包已存在：`docs/tasks/Phase2/T47_paper_ablation_result_pack_and_material_ledger.md`，且只允许 hedge-conditioned docs-only 推进。
10. `T15` 是否应直接运行多场景 P4 smoke？
   - 当前答案：已执行完成。
     - run dir: `runs/p4_benchmark/p4multis_20260508_221718_b82874_48280`
     - matrix:
       - `static_bias_theta + linear_ramp`
       - `ekf / ukf / constant_residual_mu / rls_residual_b / hybrid_residual_b`
       - `--paired-seeds`
       - `--repeats 2`
       - `C:\ProgramData\anaconda3\python.exe`
       - `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
     - two scenario winners:
       - `hybrid_residual_b`
       - `hybrid_residual_b`
11. `T15` 的 review warning 如何处理？
   - 当前答案：
     - N1 handoff 状态不一致：`accepted`，Captain 已修正 04/07 文档状态。
     - N2 `hybrid_residual_b` teacher diagnostics 全零：`T16` 已判定为非阻塞风险，继续保留在 R10。
     - N3 `delta_rows` 为 null：`accepted`，这是 strong-baseline config 不包含 `static_linear` / `cnn_fpga` 的预期后果，不应误判为缺失结果。
12. `T17` 的 review warning 如何处理？
   - 当前答案：
     - Verdict：`PASS`。
     - N1 `torch` dev build：`accepted`，只作为本机环境事实记录，不写成跨机器保证，风险保留到 R11。
     - N2 未产出 `requirements-train.txt`：`accepted`，因为任务允许用 `docs/training_chain_bootstrap.md` 收口；训练链可移植性后续单开任务。
13. `T18` 的主要结论是什么？
   - 当前答案：
     - Verdict：`PASS`。
     - N1 推荐表述 Markdown 格式问题：`accepted`，只作排版提醒，不写入 risks。
     - `.tflite` export/runtime 代码路径存在。
     - `tflite_stub_v1` 是明确的回退路径，不等于真实部署。
     - 本机未安装 `tensorflow` / `tflite_runtime`，因此真实 `.tflite` runtime 仍未恢复。
14. `T19` 的 review warning 如何处理？
   - 当前答案：
     - Verdict：`PASS`。
     - N1 preflight glob 在 PowerShell 下可能有 shell 展开差异：`accepted`，作为后续 cleanup 执行任务的命令写法注意，不写入风险升级。
     - N2 tracked `.pyc` = `116` 与工作区 `.pyc` 总数 `133` 的差异说明：`accepted`，差异来自未跟踪/忽略缓存，不影响 T19 只处理已跟踪文件的结论。
15. T20 是否可以开始？
   - 当前答案：
     - 已完成并通过 adversarial review。
     - 产物仍只是 readiness checklist，不是 real-board validation。
16. T20 当前补出的主要结论是什么？
   - 当前答案：
     - `docs/real_board_hil_readiness.md` 已形成。
     - 当前真板路径仍应标记为 `placeholder_real_board_backend`。
     - 后续真板 smoke 至少要补齐设备存在、寄存器活性、DMA histogram 读出、commit/ack round-trip 四层证据。
     - 在这些证据出现前，不得把 `board` backend、`/dev/uio*` 配置项或现有 HIL 日志写成真板完成。
17. T20 的 review warning 如何处理？
   - 当前答案：
     - Verdict：`PASS`，Captain 按 `PASS_WITH_WARNINGS` 收口。
     - N1 寄存器名来源不透明：`deferred`，后续真板执行任务必须直接审计 `axi_map.py` / DMA 代码与 RTL 地址表。
     - N2 验收标准缺量化阈值：`deferred`，后续真板 smoke plan 必须补 timeout、shape、epoch 变化与 commit/ack 阈值。
     - N3 权限描述偏 Linux：`deferred`，后续任务必须先确认目标平台是 Linux 还是 Windows，并据此更新权限/driver 模型。
18. T21 为什么不是直接真板 smoke？
   - 当前答案：
     - `T14` 至 `T20` 已完成一个 Phase 2 任务队列，应先做 milestone review。
     - 真板 smoke 还缺 R13/R14 所列设备、权限、地址表、量化阈值与平台确认。
     - 直接执行真板 smoke 可能把 readiness checklist 误当成 hardware validation。
19. T21 当前的 gate 结论是什么？
   - 当前答案：
     - `docs/review/T21_phase2_milestone_review.md` 已形成。
     - gate decision = `Conditional`。
     - 允许继续 bounded Phase 2 开发，但不升级当前证据为 formal benchmark、真实 `.tflite` runtime、physical cleanup 或 real-board validation。
20. T21 推荐的下一唯一任务是什么？
   - 当前答案：
     - 推荐下一唯一任务为 `T22: Real-board smoke execution plan with platform / AXI-map audit and quantitative acceptance thresholds`。
     - Captain 已接受该建议，并已创建 `docs/tasks/Phase2/T22_real_board_smoke_execution_plan.md`。
21. T22 是否可以直接调用真板？
   - 当前答案：
     - 不可以。`T22` 只制定 execution plan，允许只读审计源码/文档。
     - 禁止调用硬件命令、禁止运行 `backend=board` HIL、禁止修改 `board_backend.py` / `fpga_driver.py` / `run_hil_suite.py`。
     - T22 的输出不能写成 real-board validation，只能写成后续硬件任务的进入条件和执行计划。
22. T22 当前已经产出了什么？
   - 当前答案：
     - 已新增 `docs/real_board_smoke_execution_plan.md`。
     - 已补 Linux / Windows / WSL / remote board host 决策点。
     - 已补 AXI/register map 审计清单、DMA buffer 审计清单、Layer A-D 量化阈值草案、fail-fast budget 和 future evidence pack。
     - 这些产物仍然只是 plan-only，不是真板执行记录，也不是 hardware validation。
23. T22 的 review warning 如何处理？
   - 当前答案：
     - Verdict：`PASS_WITH_WARNINGS`，blocking issues: none。
     - N1 out-of-scope governance files：`accepted`，Captain 确认为 T21/T22 整合阶段的治理同步，不归为 Worker 越界。
     - N2 `AXI_REGISTER_MAP` preflight 输出为 dataclass repr：`deferred`，后续真板执行任务需要格式化地址表输出。
     - N3 `byte_count = 4096` 假设依赖 `32 x 32 float32`：`deferred`，后续真板执行任务必须用实际 bitstream / DMA contract 确认。
24. T23 为什么不是直接论文 roadmap、formal benchmark 或真板执行？
   - 当前答案：
     - 论文发表是最终目标，但当前仍要按证据等级逐步推进。
     - 当时最大软件证据缺口是 `T15` 仍未升级为 formal benchmark；现在 `T24` 已补齐 frozen-set formal software revalidation。
     - `T23` 锁定 P4 formal protocol、baseline、seed/repeat、统计报告、compute budget 和 `T24` go/no-go 条件；后续机制诊断和论文收口仍必须逐项新开任务包。
25. 新深度研究报告是否要求调整当前任务安排？
   - 当前答案：
     - 不需要推翻当前 T23；报告反而支持“先 benchmark protocol，后机制/runtime/真板”的顺序。
     - 需要增强 T23 任务包：加入报告本身和 paper-inspired 草案作为输入，并要求 Worker 明确评估强 classical / soft-information / calibration / learned baseline、更多 drift scenario、seed/CI/latency/commit/fallback 指标。
     - 需要调整后续大纲：在机制任务前补入 calibration/statcalib baseline feasibility gate；`T24` 由 T23 gate 决定是直接执行还是先补 prerequisite。
26. T24 应直接执行什么范围？
   - 当前答案：
     - 已执行历史 frozen-set 的 bounded formal software revalidation：
       - `static_bias_theta / linear_ramp / step_sigma_theta / periodic_drift`
       - `ekf / ukf / constant_residual_mu / rls_residual_b / hybrid_residual_b`
       - `paired_seeds`
       - `repeats=2`
     - 仍固定为 `mock-backed` software HIL，不是 `.tflite`、不是真板。
27. `statcalib` baseline 是否必须先于 T24 实现？
   - 当前答案：
     - 对“历史 frozen-set formal software revalidation”本身：不是硬阻塞。
     - 对更接近 paper-grade 的 benchmark 说服力：是强烈建议的后续 comparator，应保留为独立任务，不应静默塞进 T24。
28. 深度研究建议的 `random-walk / sinusoidal / burst-reset`、CI-driven stopping、soft-information comparator 是否进入 T24？
   - 当前答案：
     - 不进入 T24。
     - 这些都属于 formal-benchmark scope expansion，必须在 frozen-set revalidation 之后通过新的独立任务评估是否纳入。
29. T23 reviewer warnings 如何处理？
   - 当前答案：
     - Verdict：`PASS_WITH_WARNINGS`，blocking issues: none。
     - N1 out-of-scope governance sync：`accepted`，按 Captain 整合处理。
     - N2 exact CLI shape：`deferred`，已写入 R19，并在 T24 任务包中固定 repeat-chunked CLI。
     - N3/N4 requested metric availability：`deferred`，已写入 R19；T24 必须报告实际可用字段与缺失字段。
30. T24 是否可以直接提交给 Worker 执行？
   - 当前答案：
     - 已执行完成，并由 Captain 接受为 `PASS_WITH_WARNINGS`。
     - 当前不再提交 T24；T25 gate review 也已完成。
31. T25 是否可以直接提交给 Worker 执行？
   - 当前答案：
     - 已执行完成，不再提交给 Worker。
     - Captain verdict = `PASS_WITH_WARNINGS`。
     - T25 本身是 review 工作，本轮不启用重复 Claude review。
32. T25 当前 gate review 的结论是什么？
   - 当前答案：
     - Captain 接受 verdict = `PASS_WITH_WARNINGS`。
     - T24 可以视为 completed frozen-set formal software revalidation。
     - T24 仍只能表述为 mock-backed software HIL，不得升级为 `.tflite` runtime、`real_board` 或 paper-grade expanded benchmark。
     - `correction_saturation_rate_mean` structural zero 继续保留在 R20。
     - `teacher_scalar_diagnostics.csv` header-only / teacher diagnostics 全零继续保留在 R10。
33. T25 之后当前最推荐的下一类任务是什么？
   - 当前答案：
     - Captain 接受 T25 推荐：`T27: Teacher diagnostics path audit and mechanism-evidence repair plan`。
     - 理由是 R10 的 deferred 链最长，且它已经开始影响对 `hybrid_residual_b` 机制解释的可信度。
     - `T27` 只读审计路径和产出修复计划，不运行新 benchmark、不改源码、不补新 baseline。
34. T26 是否可以作为下一任务直接交给 Worker？
   - 当前答案：
     - 可以，现在已切换为当前唯一任务。
     - 但 T26 只能做 docs-only/read-only feasibility gate 和最小设计计划，不得实现 statcalib、不运行 benchmark、不改 formal benchmark 口径。
35. T27 当前 path audit 的结论是什么？
   - 当前答案：
     - Captain 接受 verdict = `PASS_WITH_WARNINGS`。
     - R10 主因已缩窄：broadcast teacher features 不会触发当前 scalar explain diagnostics；hybrid path 是 `data not generated`，不是 writer 单点漏写。
     - R20 不共享 teacher diagnostics 死路径；当前 T24 零值更像当前参数区间下未触发 saturation。
     - downstream CSV `0.0` coercion 是独立 missing-vs-zero 语义风险，写入 R21。
36. T28 是否可以交给 Worker？
   - 当前答案：
     - 已执行完成，并由 Captain 接受为 `PASS_WITH_WARNINGS`。
     - T28 smoke 已验证 missing-vs-zero 输出语义。
37. T28 reviewer warnings 如何处理？
   - 当前答案：
     - N1 duplicate markdown report header row：`deferred`，写入 R22，并作为 T29 当前唯一任务。
     - N2 tracked `.pyc` side-effect：`rejected as technical signal`，不作为有意义改动提交。
     - N3 `comparison.csv` column order changed：`accepted`，属于 T28 语义修复的预期接口变化。
     - Missing focused tests：`deferred`，写入 R23。
     - S1/S2/S3：`accepted`，符合当前修复语义。
38. T29 是否可以交给 Worker？
   - 当前答案：
     - 已执行完成，Captain verdict = `PASS`。
     - N1 tracked `.pyc` side-effect 按 known repo-noise / rejected technical signal 处理，不作为技术改动提交。
39. T26 是否可以提交给 Worker 推进？
   - 当前答案：
     - 不再提交；T26 已完成并通过 Captain `PASS` 收口。
     - T26 的 follow-up 是 T30：只允许收紧 interface contract 与 separate comparator lane 最小实现边界，不得运行 benchmark、新增 formal run dir、改 formal protocol、触碰 `.tflite` 或真板路径。
40. T30 是否可以提交给 Worker 推进？
   - 当前答案：
     - 不再提交；T30 已完成并通过 Captain `PASS` 收口。
     - T30 的 output 是 interface-only statcalib contract 和 focused tests，不是 slow-loop integration 或 formal benchmark evidence。
41. T30 reviewer warnings 如何处理？
   - 当前答案：
     - N1 gate doc stale non-claim：`accepted`，Captain 已修正 `docs/statcalib_feasibility_gate.md`。
     - N2 `tests/` 无 `__init__.py`：`accepted`，当前 unittest 发现机制足够；后续测试目录增长时再整理。
     - N3 `tests/__pycache__` side-effect：`rejected as technical signal`，不作为有意义技术改动提交。
     - N4 residual-b baseline assumption：`deferred`，已写入 R24。
42. T36 是否可以交给 Worker？
   - 当前答案：
     - 不再提交；T36 已完成并通过 Captain `PASS` 收口。
     - T36 已读取既有 `runs/teachrepr*` 结果并产出 `docs/seed20260429_failure_diagnosis.md` 与只读分析脚本。
     - T36 结论：`20260429` 更像 residual-amplitude / teacher-delta regime instability，但缺少 per-window trace，不能证明 sign offset、overshoot chronology 或 teacher-vs-CNN attribution。
43. T36 reviewer warnings 如何处理？
   - 当前答案：
     - Verdict：`PASS`。
     - N1 unused `Iterable` import：`accepted` as cosmetic。
     - N2 hardcoded folder mappings：`accepted`，因为该脚本是 bounded frozen-artifact diagnostic，不是 reusable production tool。
     - N3 worker pre-review file 被 adversarial review 覆盖：`accepted`，Worker verification 已保留在任务包。
44. T38 是否可以交给 Worker？
   - 当前答案：
     - 可以，但只能做 `seed=20260429` single-seed trace-export probe。
     - 允许一个 T38-scoped bounded rerun，用于导出 per-window `teacher_b`、predicted `delta_b`、committed `b` 和 window outcome/utilization。
     - 禁止训练、扩 teacher-representation 分支、新增 baseline/scenario、改 formal benchmark protocol、触碰 `.tflite` 或真板路径。

## 暂缓事项

以下事项重要，但在新的任务包明确前暂缓：

1. `noise_channels -> effective parameters` 桥接
2. load-aware latency injector
3. stateful fault injector
4. bit-accurate control pipeline
5. teacher-representation 新分支扩展或长跑
6. 未经 `T14` 审计的 P4 长跑或正式 benchmark
7. 未经新任务包批准的 P4 剩余场景补跑
8. statcalib slow-loop integration 或 formal benchmark integration

## 2026-05-16 Captain Update

45. T38 review 如何裁决？
   - 当前答案：
     - Captain verdict = `PASS`。
     - N1/N2/N3/N4 全部归类为 `accepted`。
     - 没有 `deferred` warning，因此未从 T38 warning 分类新增 risk。
46. Milestone 2I 是否允许进入下一里程碑？
   - 当前答案：
     - `docs/review/Milestone2I_review.md` verdict = `Conditional Allow`。
     - 允许进入下一 bounded milestone，但不允许把 T38 写成 full causal proof、mitigation success、clean-env proof、runtime validation 或 real-board validation。
47. 当前下一唯一任务是什么？
   - 当前答案：
     - `T31: Training-chain portable dependency lock plan`。
     - 任务包为 `docs/tasks/Phase2/T31_training_chain_portable_dependency_lock_plan.md`。
     - T31 只做 clean-environment / dependency-lock plan，不安装依赖、不训练、不运行 benchmark、不创建 `runs/` 或 `artifacts/`。

### Risk Status Update

- `R10` remains open but narrowed: T38 provides trace-level evidence for seed=20260429, but does not provide mitigation, multi-seed confirmation, or upstream root-cause isolation.
- `R11` remains open but further narrowed: T40 completed one clean-environment real-training smoke, but this still does not prove full training reproducibility or broader portability.
- `R20`, `R23`, and `R24` remain valid and are not closed by T38 or the Milestone 2I review.

## 2026-05-17 Captain Update

48. T31 review 如何裁决？
   - 当前答案：
     - Captain verdict = `PASS`。
     - Blocking issues: none。
     - N1 markdown subsection numbering：`accepted` as cosmetic。
     - N2 later alignment with `docs/training_chain_bootstrap.md`：`accepted` as future alignment。
     - N3 worker self-review overwritten by adversarial review：`accepted`。
     - 没有 `deferred` warning，因此未从 T31 warning 分类新增 risk。
49. T40 review 如何裁决？
   - 当前答案：
     - Captain verdict = `PASS`。
     - Blocking issues: none。
     - N1 worker pre-review overlap：`accepted`。
     - N2 legacy macOS dataset-manifest paths：`accepted`。
     - N3 R11 governance sync：`deferred`，并已写回当前治理同步。
50. T40 是否关闭 R11？
   - 当前答案：
     - 不关闭。
     - T40 已把 R11 从“只有 draft lock/dry-run/import”缩窄为“clean environment 已完成 one real-training smoke”。
     - full training reproducibility、GPU/CUDA portability、Linux portability 仍未验证。
51. 当前下一唯一任务是什么？
   - 当前答案：
     - `T35: Paper draft skeleton and reviewer-risk audit`。
     - 任务包为 `docs/tasks/Phase2/T35_paper_draft_skeleton_and_reviewer_risk_audit.md`。
     - T35 只允许做 docs-only paper skeleton 与 reviewer-risk audit；不得运行新实验、不得升级 evidence level、不得改写阶段结论或 repo facts。

52. T35 review 如何裁决？
   - 当前答案：
     - Captain verdict = `PASS`。
     - Blocking issues: none。
     - N1 title candidates are unusually conservative：`accepted`。
     - N2 skeleton omits Background / Related Work section：`accepted`。
     - N3 section-by-section hotspot table uses generic labels：`accepted`。
     - N4 worker pre-review overwritten by adversarial review：`accepted`。
     - 没有 `deferred` warning，因此未从 T35 warning 分类新增 risk。
53. Milestone 2K 是否已经完成？
   - 当前答案：
     - 是。`T34 + T35` 都已完成并通过 Captain `PASS` 收口。
     - 但这只代表 paper-assembly readiness 已到位，不代表可以跳过 paper-positioning gate、也不代表 blocked evidence 已升级。
54. 当前下一唯一任务是什么？
   - 当前答案：
     - `T41: Milestone 2K paper-assembly gate review and next-phase decision`。
     - 任务包为 `docs/tasks/Phase2/T41_paper_assembly_milestone_review.md`。
     - T41 只允许做 read-only milestone review；不得运行新实验、不得升级 evidence level、不得改写阶段结论或 repo facts。
55. T41 review 如何裁决？
   - 当前答案：
     - Captain verdict = `PASS`。
     - Blocking issues: none。
     - N1 T34 review path typo：`accepted`，并已在 Captain integration 中修正。
     - N2 T41 human explanation count typo：`accepted`，并已在 Captain integration 中修正。
     - 没有 `deferred` warning，因此未从 T41 warning 分类新增 risk。
56. Milestone 2K 当前状态是什么？
   - 当前答案：
     - 已正式关闭，gate verdict = `Allow`。
     - 但这不代表 blocked evidence 已升级；它只代表 paper-assembly readiness 已可进入下一步结构扩展。
57. 当前下一唯一任务是什么？
   - 当前答案：
     - `T42: Paper Background / Related Work scaffold and method-positioning calibration`。
     - 任务包为 `docs/tasks/Phase2/T42_paper_background_related_work_and_positioning.md`。
     - T42 只允许做 docs-only 结构扩展与定位校准；不得运行新实验、不得升级 evidence level、不得改写阶段结论或 repo facts。
## 2026-05-24 Captain Update (T47 closeout supersession)

- `T47` review 已由 Captain 接受为 `PASS`。
- Warning classification:
  - N1 figure-entry miscount = `accepted`
  - N2 worker-summary boundary drift = `accepted`
  - N3 F2 ready-without-drawn-figure classification = `accepted`
  - N4 F3 blocked carry-forward note = `accepted`
  - N5 conceptual regeneration paths instead of executable scripts = `accepted`
- 没有 `deferred` 或 `rejected` warning，因此这次不新增由 warning classification 触发的 risk。
- `T47` 已完成，但 `FR7` 仍是最大的显式 ablation gap。
- 当前唯一任务已切换为 `T57: FR7 feature/teacher ablation re-execution under locked T24 protocol`，任务包为 `docs/tasks/Phase2/T57_fr7_feature_teacher_ablation_reexecution.md`。

## 2026-05-26 Captain Update (T58 closeout supersession)

- `T58` review 已由 Captain 接受为 `PASS_WITH_WARNINGS`。
- Warning classification:
  - `N1` = `accepted`
  - `N2` = `accepted`
  - `N3` = `accepted`
  - `N4` = `accepted`
- 没有新的 `deferred` 或 `rejected` warning，因此本次不新增由 warning classification 触发的 risk。
- `T58` 关闭的是 `FR6` 的 bounded figure-pack 缺口，不是 `R10` 的 causal closure，也不是 `C4` 的支持级升级。
- 当前 mainline 最大缺口已切换为 `FR8 / statcalib integrated comparator result table`，而当前最小下一步不是直接写 `FR8`，而是先做 `T59` 的 separate comparator lane integration + bounded smoke。
- 当前唯一任务切换为 `T59: Statcalib separate comparator lane integration and bounded smoke`，任务包为 `docs/tasks/Phase2/T59_statcalib_comparator_lane_integration_and_smoke.md`。
- `R24` 继续有效：在 `T59` 之前和之后，都不得把 `statcalib` interface/helper 或 smoke lane 外推为已验证的正式 comparator evidence。

## 2026-05-26 Captain Update (T57 closeout supersession)

- `T57` review 已由 Captain 接受为 `PASS`。
- `T57` review 没有新的 blocking issue，也没有新的 non-blocking item 需要再做 `accepted / deferred / rejected` 分类，因此不会新增由 warning classification 触发的 risk。
- `FR7` 现在可以作为 bounded frozen-set ready result table 收口，但 `R10` 仍然开放；`T57` 不是 causal proof，不是 mechanism closure，也不是 expanded benchmark evidence。
- 当前最强的 paper-risk 边界是：`hybrid_no_teacher_params` 在 4 个 frozen scenarios 中都成为最佳模式，因此 teacher-parameter necessity 仍然不能宣称成立。
- 当前最大的 paper-material gap 已切换为 `FR6`；`FR8` 仍然排在后续。
- 当前唯一任务切换为 `T58: FR6 multi-seed mechanism/intervention figure pack`，任务包为 `docs/tasks/Phase2/T58_fr6_multi_seed_mechanism_intervention_figure_pack.md`。
- `T58` 是 docs-only，必须复用既有 `T54/T55/T56` 证据，不得启动新的 benchmark、trace、intervention、`.tflite`、real-board、training、cleanup 或 theory-branch 工作。

## 2026-05-26 Captain Update (T59 closeout supersession)

- `T59` review has been accepted by Captain as `PASS_WITH_WARNINGS`.
- Warning classification:
  - `W1` cross-mode `teacher_mode` fallback coupling = `deferred` -> `R26`
  - `W2` smoke-doc key-name mismatch = `accepted`
  - `W3` dirty-worktree smoke provenance weakness = `deferred` -> `R27`
- Additional carry-forward concerns from the review's missing-tests and suspicious-details sections are now covered by `R26` and `R27` rather than treated as closed.
- `T59` closes separate-lane integration and one bounded smoke only. It does not open `FR8`, and it does not close `R24`.
- The current unique task is now `T60: Statcalib lane isolation and regression hardening`, task package `docs/tasks/Phase2/T60_statcalib_lane_isolation_and_regression_hardening.md`.
- `R26` and `R27` must be treated as pre-FR8 blockers.

## 2026-05-27 Captain Update (T60 closeout supersession)

- `T60` review has been accepted by Captain as `PASS`.
- `T60` review introduces no new warning item that needs `accepted / deferred / rejected` handling.
- `T59` warning `W1` is now closed by `T60`; `R26` should now be treated as closed.
- `R27` remains open but narrower: T60 closes the regression-coverage gap, while provenance-clean fairness/robustness sanity is still missing before any `FR8` task.
- `T60` closes semantics/test hardening only. It does not open `FR8`, and it does not close `R24`.
- The current unique task is now `T61: Statcalib clean-provenance fairness sanity rerun`, task package `docs/tasks/Phase2/T61_statcalib_clean_provenance_fairness_sanity.md`.
- `T61` remains a pre-FR8 blocker-clearing task only; it must not be rewritten into formal comparator evidence, `.tflite` validation, or real-board validation.

## 2026-05-27 Captain Update (T61 closeout supersession)

- `T61` review has been accepted by Captain as `BLOCK`.
- `T61` preserved the bounded fairness signal, but it did not close the clean-provenance blocker it was created to repair.
- `R27` therefore remains open with concrete evidence from `T61`: clean launch `HEAD=9174065`, final `summary.json git_commit=6058f42`, and mid-run branch movement mean the run still lacks one defensible commit identity.
- No new warning-derived risk item is opened from T61 because the verdict is `BLOCK`, not `PASS_WITH_WARNINGS`.
- The current unique task is now `T62: Statcalib provenance-isolated fairness rerun`, task package `docs/tasks/Phase2/T62_statcalib_provenance_isolated_fairness_rerun.md`.
- `T62` is the single automatic retry for the same blocker. If `T62` still returns `BLOCK`, Captain should stop automatic progression and return the issue to the user.

## 2026-05-27 Captain Update (T62 closeout supersession)

- `T62` review has been accepted by Captain as `PASS`.
- `T62` review introduces no new warning item that needs `accepted / deferred / rejected` handling.
- `T62` closes the specific provenance blocker that caused `T61` to fail, so `R27` should now be treated as closed.
- `T62` still does not open `FR8`, and it does not close `R24`.
- The current unique task is now `T63: FR8 statcalib comparator gate review`, task package `docs/tasks/Phase2/T63_fr8_statcalib_comparator_gate_review.md`.
- `T63` is docs-only and exists to decide whether a bounded FR8 task should exist at all, not to start FR8 automatically.

## 2026-05-27 Captain Update (T63 closeout supersession)

- `T63` review has been accepted by Captain as `PASS`.
- `T63` review introduces no new warning item that needs `accepted / deferred / rejected` handling.
- No new risk item is opened by warning classification for `T63`.
- `R27` remains closed by `T62`.
- `R24` remains open, but after `T63` it should be treated as the main scope/reporting constraint on `T64`, not as a blocker that requires another pre-FR8 prerequisite task.
- The current unique task is now `T64: FR8 statcalib extension-lane bounded benchmark`, task package `docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md`.
- `T64` must keep `statcalib` as a separately labeled extension lane, preserve the historical frozen ranked table, and remain inside mock-backed software-HIL scope only.

## 2026-05-29 Captain Update (T65 closeout supersession)

- `T65` review has been accepted by Captain as `PASS_WITH_WARNINGS`.
- Warning classification:
  - `N1` mixed-diff scope acceptance depends on explicit user/captain clarification = `accepted`
  - `N2` T64-specific audit helper is intentionally narrow, not generic FR8 framework = `accepted`
  - `N3` review wording should have stated the clarification dependency more explicitly = `accepted`
- No new risk item is opened by warning classification for `T65`.
- `R28` should now be treated as closed by `T65`: the T64 report pack is now backed by an explicit audit helper, focused tests, and a bounded audit document.
- `R24` remains open and is now the dominant mainline comparator-scope risk.
- The current unique task is now `T66: FR8 statcalib sensitivity bounded benchmark`, task package `docs/tasks/Phase2/T66_fr8_statcalib_sensitivity_bounded_benchmark.md`.
- `T66` exists to answer whether the T64 statcalib win is robust across a small predeclared heuristic grid, without rewriting `T24` or upgrading the evidence beyond mock-backed software-HIL scope.

## 2026-06-01 Captain Update (T66 closeout supersession)

- `T66` review has been accepted by Captain as `PASS_WITH_WARNINGS`.
- Warning classification:
  - `N1` duplicate-running progress-log artifact after same-run-root timeout relaunch = `accepted`
  - `N2` aggregate-best vs stability-best split = `deferred -> R24`
  - `N3` `static_bias_theta / statcalib_high_threshold` best row still carries aggregate `statcalib_status = mixed` = `deferred -> R24`
- No new standalone risk item is opened by warning classification for `T66`.
- `R24` remains open, but it is now more specific: the dominant unresolved question is teacher-anchor dependence, not local-parameter fragility.
- The current unique task is now `T67: FR8 statcalib teacher-anchor dependence bounded benchmark`, task package `docs/tasks/Phase2/T67_fr8_statcalib_teacher_anchor_dependence_bounded_benchmark.md`.
- `T67` exists to answer whether the strongest T66 statcalib points remain competitive when `teacher_mode` changes, without rewriting `T24` or upgrading the evidence beyond mock-backed software-HIL scope.
