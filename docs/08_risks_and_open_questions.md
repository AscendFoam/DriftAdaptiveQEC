# Risks And Open Questions

## 风险清单

| ID | Risk | Level | Evidence | Mitigation |
| --- | --- | --- | --- | --- |
| R1 | 默认运行环境不可直接执行最小 benchmark | 中 | 默认 `python 3.13.7` 仍缺 `numpy`，但 `C:\ProgramData\anaconda3\python.exe` 已可跑通 P0 smoke | 后续所有治理文档继续显式指定推荐解释器 |
| R2 | 根目录虽已补 recovery-scoped manifest，但完整训练链、`.tflite` 与真板环境仍无统一依赖说明 | 中 | `requirements-recovery.txt` 只覆盖 `P0/P3/P4 recovery smoke`，且显式不含 `torch`、`tensorflow`、`tflite-runtime`；`docs/training_chain_bootstrap.md` 已补训练链 bootstrap，但还不是跨机器完整依赖锁定 | 继续保持作用域诚实；训练链已独立说明，`.tflite` 与真板路径仍需单开有界 manifest / bootstrap 任务 |
| R3 | 软件 HIL 与真板 HIL 边界容易被误写 | 高 | `cnn_fpga/hwio/board_backend.py` 仍是 placeholder 风格；`docs/03_hil_p4_boundary_audit.md` 已完成边界澄清；`T20` 当前任务只允许补 readiness checklist，不允许实现或宣称真板完成 | 后续所有文档、复验与报告都必须引用 `docs/03_hil_p4_boundary_audit.md` 的统一口径；`T20` 只做只读 readiness 审计 |
| R4 | 仓库中已有大量缓存与生成物噪声 | 中 | `.gitignore` 已忽略 `__pycache__/`、`runs/`、`artifacts/`，但 Git 中仍有 `116` 个已跟踪缓存/字节码文件、`1841` 个已跟踪 `runs/` 文件、`110` 个已跟踪 `artifacts/` 文件；`T19` 已确认这 `116` 个文件全部位于 `9` 个 `__pycache__` 目录中 | 已补 `docs/06_repo_noise_governance.md` 与 `docs/cleanup_tracked_cache_manifest.md` 固定“先治理后清理”；后续仍需单开 cleanup 执行任务做物理移除 |
| R5 | P4 目前已完成四场景、五模式、`repeats=2` 的 formal software revalidation，但仍为 mock-backed software HIL，不是 `.tflite` runtime、不是 `real_board`、不是 paper-grade expanded benchmark | 中 | T24 run dir: `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`；`missing_runs = []`，`coverage = 1.0`；四场景 winner 均为 `hybrid_residual_b` | T24 已完成 formal frozen-set software revalidation；但仍不可写成 paper-grade benchmark 或 runtime/board validation；后续如要升级证据等级，需新增 statcalib comparator、CI-driven stopping 或 runtime/board 路径 |
| R6 | `.tflite` 真导出与 stub 回退容易混淆 | 中高 | `cnn_fpga/model/export.py` 与 `cnn_fpga/runtime/inference_service.py` 同时支持两种路径，且 runtime 输出不同 `source`；`T4/T7` 当前都刻意未走 `.tflite` 路径 | 文档与日志必须显式标注 `artifact type`，并区分 `tflite_service` 与 `tflite_stub_service` |
| R7 | 虽然 `T5` 已立治理口径，`T19` 也已补出缓存 cleanup manifest，但具体 cleanup 执行窗口与归档方式仍未决定 | 中 | `docs/06_repo_noise_governance.md` 与 `docs/cleanup_tracked_cache_manifest.md` 已固定缓存 cleanup 的目标目录、命令草案、回滚方式与验收标准，但尚未执行物理 cleanup | 在后续单开有界 cleanup 执行任务，严格按 manifest 落地，并继续把 `runs/` / `artifacts/` 留在独立任务中处理 |
| R8 | 最小 software HIL 路径虽然已在 bounded recovery path 上完成逐字一致复验，但该结论容易被误外推到真板、`.tflite` 或正式 benchmark | 中 | `T12` 已确认 `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104` 与 `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104` 的 `hil_summary.json` / `hil_events.json` 哈希一致；但路径仍固定为 `mock + model_artifact + artifact_npz + inproc` | 后续文档必须继续写清结论边界，不把 bounded recovery smoke 扩写成真板或正式 benchmark 已恢复 |
| R9 | T24 已完成 formal frozen-set revalidation，但若继续扩大到更多 repeat、CI-driven stopping 或 extra drift families，仍可能隐式越过 frozen-set/formal 边界 | 中 | T24 已按 locked protocol 跑完 `4 scenarios x 5 modes x repeats=2`；`docs/P4_benchmark_formal_protocol.md` 已锁定边界 | T24 后仍不应自动追加更大 repeat、额外 scenario 或 statcalib comparator；任何进一步 P4 扩展都必须新开任务包 |
| R10 | `hybrid_residual_b` 的 teacher diagnostics / mechanism evidence 仍缺 per-window trace，因此机制解释尚不能闭环 | 中 | `docs/review/T27_teacher_diagnostics_path_audit.md` 已将主因缩窄为 broadcast teacher features 不触发 scalar explain diagnostics；`docs/review/T28_review.md` 确认当前输出已显式标记 `not_generated` / `not_applicable`；`docs/seed20260429_failure_diagnosis.md` 将 `20260429` 收益收缩缩窄为 residual-amplitude / teacher-delta regime instability hypothesis，但缺少 per-window committed-parameter trace | R10 remains open but further narrowed；T38 已开任务包，后续只允许 single-seed trace-export probe，不得扩 benchmark 或新分支 |
| R11 | 训练链已有 portable dependency-lock plan，但 clean-environment CPU lock 尚未实际创建或验证 | 中 | `docs/training_chain_portable_dependency_lock_plan.md` 已由 T31 产出并通过 `docs/review/T31_review.md` `PASS`；计划确认 CPU-only lane 合理、`DLEnv` / dev torch 只属于 local evidence，但 T31 未创建 clean environment、未产出 real lockfile、未运行 dry-run bootstrap | 不把 T31 plan 写成 clean-env proof；下一 bounded task `T39` 只允许创建 CPU-only draft lock 和 dry-run/import-level bootstrap，不运行训练或 benchmark |
| R12 | `.tflite` 路径已有代码与入口，但真实 TensorFlow / TFLite 运行时在当前机器上不可用 | 高 | `docs/TFLite_runtime_bootstrap.md` 已记录 `tensorflow = False`、`tflite_runtime = False`；`export.py`、`evaluate_tflite.py`、`validate_export.py` 入口存在，但真实 runtime 需独立环境 | 继续把真实 `.tflite`、stub manifest 与 HIL benchmark 边界写清；若后续要跑真实 runtime，单开环境任务或在具备依赖的机器上做独立 smoke |
| R13 | 真板 HIL 入口存在配置骨架，但距离可执行真板 smoke 仍缺设备、权限、寄存器一致性与日志证据 | 高 | `board_backend.py` 仍是 placeholder；设备缺失时会触发 `board_device_missing:...`；`schedule_commit(...)` 仍返回 `target_bank=None`、`version=None`、`ack_delay_us=None`；`step(...)` 返回空事件；`docs/real_board_hil_readiness.md` 已固定前置条件与验收标准 | 后续若推进真板路径，必须单开执行任务，逐层补齐设备存在、寄存器活性、DMA 读出与 commit/ack round-trip 证据，在此之前禁止写成 real-board HIL 已完成 |
| R14 | T22 已把寄存器来源、DMA 审计清单和量化阈值草案具体化，但真实宿主、bitstream 与 DMA contract 仍未验证 | 中高 | `docs/real_board_smoke_execution_plan.md` 已直接映射 `axi_map.py` / `dma_client.py`，`docs/review/T22_review.md` 确认 AXI/DMA 审计清单与源码吻合；但 N2 指出 preflight 输出格式仍需改进，N3 指出 `byte_count = 4096` 依赖 `32 x 32 float32` 假设 | 后续若进入真板执行任务，必须先选择宿主模型，再用实际 bitstream / RTL / DMA contract 确认地址表、histogram shape、element dtype、timeout 与 commit/ack 阈值 |
| R15 | Phase 2 当前已完成一轮 milestone queue，但证据仍混合停留在 development / bootstrap / manifest / readiness 层，若直接升级到 formal benchmark、真实 `.tflite` runtime、physical cleanup 或 real-board validation，容易再次打破边界诚实 | 高 | `docs/review/T21_phase2_milestone_review.md` verdict = `Conditional`；`T15` 仍只是 `development_smoke`；`T18` 真实 runtime 不可用；`T19` 未执行 cleanup；`T20` 不是真板验证 | 保持 `Phase 2: Controlled Development` / `Go`，继续只开 bounded 下一任务；优先补 `T22` 这类 execution-plan 级文档任务，而不是直接进入高风险执行任务 |
| R16 | 把“最终要发论文”误压缩成最近任务直接写论文 claim，容易跳过 formal benchmark、机制诊断和部署边界证据 | 高 | `T24` 已完成 frozen-set formal software revalidation，`T25` 已确认其 result boundary，`T27/T28` 已缩窄并修复 teacher diagnostics 输出语义，`T29` 已修复人读 report header，`T26` 已完成 statcalib feasibility gate；但 `T18` 未恢复真实 `.tflite` runtime，`T22` 不是 hardware validation，R10 仍不是完整机制证据 | Paper claim/evidence ledger 仍应推迟到机制证据更清楚之后；当前 `T30` 只做 statcalib interface contract / bounded implementation package，不写论文 claim |
| R17 | 深度研究报告建议的 formal benchmark 范围可能显著扩大，若无分级采纳会把 T23 变成不可执行的大任务 | 中高 | `docs/reference/进一步的深度研究结果.md` 建议加入强 classical / soft-information / calibration / learned baseline 类别、更多 drift families、训练/评测 seed 分离、置信区间、latency/commit/rollback 指标和 statcalib baseline | T23 只做 protocol lock：必须把建议分类为 adopted / deferred / rejected，并通过 go/no-go 判断 T24 是执行 formal run 还是先补 prerequisite |
| R18 | `T24` 已按 frozen-set scope 完成，但若后续把 `statcalib`、soft-information、额外 drift families、CI-driven stopping、`.tflite` runtime 或真板边界并入同一任务，仍会重新打破 scope | 中高 | `docs/P4_benchmark_formal_protocol.md` 已把 `T24` gate 锁为 `GO_FOR_BOUNDED_FORMAL_SOFTWARE_REVALIDATION` + `NO_GO_FOR_SCOPE_EXPANSION_INSIDE_T24`；T24 实际完成 matrix 为 `4 scenarios x 5 modes x 2 repeats`；T25 已接受该边界；T26 gate verdict = `CONDITIONAL_GO` for separate comparator lane only | `T30` 只允许收紧 statcalib concrete interface contract 与 separate comparator lane 最小实现边界；不得把 comparator 并入 frozen ranked set，不得扩展 soft-information、额外 scenario family、true `.tflite` 或真板边界 |
| R19 | T24 formal execution 已固定 exact CLI 和报告了 metric availability | 已收口 | T24 已使用 repeat-chunked CLI shape，所有请求统计字段已存在于 `comparison.csv`；`correction_saturation_rate_mean` 全零、teacher diagnostics 全零已报告为缺口 | R19 已由 T24 Worker 收口；后续若 runner 更新指标路径，需重新验证 |
| R20 | `correction_saturation_rate_mean` 在 T24 所有 20 个 scenario/mode rows 中结构性为 0.0；T27 已证明它不共享 teacher diagnostics 死路径，但尚未证明所有参数区间都不会触发 | 中 | `docs/review/T27_teacher_diagnostics_path_audit.md` 指出该字段来自 `fast_loop_emulator.py` 独立 saturation counter，并由 HIL summary 转抄到 `comparison.csv`；当前 T24 更像现参数区间下 genuine zero | R20 remains open but materially narrowed；不在 T28 中扩大 stress run，后续如需证明触发性应单开 edge/stress 任务 |
| R21 | Teacher diagnostics downstream missing-vs-zero writer 语义已由 T28 修复 | 已收口 | `docs/review/T28_review.md` 确认 T28 smoke 中 `ukf` 为 `not_applicable`、`hybrid_residual_b` 为 `not_generated`，missing numeric teacher diagnostics 保持 empty/null，`correction_saturation_rate_mean = 0.0` 保持为独立 observed zero | R21 对当前 writer 语义关闭；未来若再次改 aggregation/report writer，应保留 `not_generated` / `not_applicable` / `true zero` 区分 |
| R22 | T28 后 `_write_report()` markdown report 存在重复 header row，导致人读 report 表格列数不一致 | 已收口 | `docs/review/T29_review.md` verdict = `PASS`；旧 11-column header 已删除；验证得到 `header_rows=1`、`column_counts=[12, 12, 12]` | R22 已由 T29 收口；未来若再改 aggregation/report writer，应按 R23 补 focused test 或静态 report-shape check |
| R23 | Aggregation/report writer 缺少 focused unit/static tests，未来可能再次出现格式或 null-semantics 回归 | 中 | `docs/review/T28_review.md` Missing Tests 指出相关路径没有现成 tests；T28 依赖 py_compile 和 bounded smoke 验证 | T28 可接受；后续再改 aggregation/report writer 时应补 focused unit test 或静态 report-shape check |
| R24 | T30 的 `from_delta_b()` 只是最小 residual-b interface helper，未来若把它误当完整 statcalib/calibration comparator，可能把接口 contract 外推成未验证算法能力 | 中 | `docs/review/T30_review.md` N4 指出当前 baseline 逻辑为 `prior.b + delta_b`，适合 residual-b comparator contract，但不是完整 calibration logic；T30 只做 interface-level tests | 后续 statcalib slow-loop integration 或 benchmark task 必须重新验证 calibration objective、fallback semantics、status propagation 和 ranking boundary，不得直接把 T30 helper 写成 validated statcalib comparator |

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
- 当前唯一任务：`T39: Training-chain CPU-only clean-environment draft lock and dry-run bootstrap`，任务包 `docs/tasks/Phase2/T39_training_chain_cpu_cleanenv_draft_lock.md`。
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
     - 当前唯一任务为 `T39` training-chain CPU-only clean-environment draft lock and dry-run bootstrap，任务包已存在：`docs/tasks/Phase2/T39_training_chain_cpu_cleanenv_draft_lock.md`。
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
- `R11` remains open but narrowed: T31 produced a portable dependency-lock plan, but the CPU-only clean environment and draft lock still need T39-style dry-run verification.
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
49. T31 是否关闭 R11？
   - 当前答案：
     - 不关闭。
     - T31 已把 R11 从“只有本机 bootstrap notes”缩窄为“已有 portable dependency-lock plan”。
     - clean Python `3.12` CPU environment、draft lock artifact、dry-run bootstrap 仍未实际验证。
50. 当前下一唯一任务是什么？
   - 当前答案：
     - `T39: Training-chain CPU-only clean-environment draft lock and dry-run bootstrap`。
     - 任务包为 `docs/tasks/Phase2/T39_training_chain_cpu_cleanenv_draft_lock.md`。
     - T39 不得运行训练、benchmark、`.tflite`、真板、cleanup 或 GPU/CUDA portability work。
