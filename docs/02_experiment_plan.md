# DriftAdaptiveQEC 实验规划与后续开发计划

**最后更新：** 2026-06-11  
**当前阶段：** `Phase 2: Controlled Development`  
**当前决策状态：** `Go`  
**当前唯一任务：** 以 `docs/04_task_board.md` 和 `docs/07_handoff.md` 为准；截至本次整理为 `T72: Real-board transfer-pack provenance hardening`。

## 文档角色

本文档现在承担两个职责：

1. **Part I：项目从开始至今的规划与证据演进**  
   只保留高层时间线、P0-P4 / T 系列关键转折、仍有效的结论，以及已被后续任务替换或降级的旧结论。
2. **Part II：后续开发计划**  
   吸收 `docs/follow-up_plan/README.md` 的功能，作为后续开发、论文准备、任务候选池和计划维护的唯一入口。

本文档不是结果证明文件。任何结果 claim 必须回到对应的 task package、review、run root、artifact、summary helper 或治理文档中验证。当前任务状态仍以 `docs/04_task_board.md` 和 `docs/07_handoff.md` 为权威来源。

---

# Part I：项目从开始至今的规划与证据演进

## 1. 项目核心问题

本项目研究 **CNN + FPGA 快慢回路协同的近似 GKP 漂移自适应解码**。

核心技术合同保持不变：

- fast loop 执行低延迟线性/仿射修正：`Delta = K @ s + b`
- slow loop 从 syndrome histogram、teacher estimate、compact statistics 或 calibration module 中产生 bounded update
- runtime 真正消费的是 `(K, b)` 或 residual/control correction，而不是离线指标里的抽象参数拟合
- `ParamMapper` 的主线语义、benchmark 口径、baseline 集合和 evidence level 不得在同一任务中被静默修改

当前最稳的论文/项目表述是：

> A deployment-bounded, teacher-anchored residual calibration framework for drift-adaptive GKP decoding under dual-loop runtime constraints.

这不是“CNN 全面替代 GKP 解码器”的项目，也不是“真实 FPGA 系统已经完成验证”的项目。

## 2. 当前证据边界快照

| 证据层 | 当前状态 | 权威锚点 | 不得外推为 |
| --- | --- | --- | --- |
| Recovery decision | `Phase 2: Controlled Development` / `Go` | `T13` recovery exit review；`docs/04_task_board.md` | 任意扩范围开发 |
| P0/P3/P4 recovery smoke | 已恢复最小可复验入口 | `docs/recovery_bootstrap/P0_smoke_bootstrap.md`、`docs/recovery_bootstrap/P3_software_hil_bootstrap.md`、`docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md` | 完整训练链、真实 `.tflite`、真板 |
| P3 software HIL | bounded recovery path 逐字一致复验 | `T12`；`mock + model_artifact + artifact_npz + inproc` | 真板 HIL 或 `.tflite` HIL |
| P4 frozen-set formal software revalidation | 已完成，历史主结果锚点 | `T24` / `T25`；`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743` | paper-grade expanded benchmark、真实 runtime、真板 |
| Mechanism evidence | 已有 multi-seed trace 与 bounded intervention，但仍不是因果闭环 | `T46`、`T54`、`T55`、`T56`、`T57`、`T58` | 简单 causal proof |
| Training/material regeneration | 已有 bounded pack 和 CPU-only rerun | `T31`、`T39`、`T40`、`T50` | full reproducibility 或跨主机保证 |
| True `.tflite` runtime | current-host isolated path 窄确认 | `T48` | 默认环境恢复、HIL closure、deployment closure |
| Real-board gate | current-host verdict 仍为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE` | `T49`、`T71`、`T72` | 真板执行成功 |
| `statcalib` | bounded mock-backed software-HIL extension lane，明确 no-promotion | `T64`-`T70` | 成熟主线 comparator 或 T24 替代表 |
| Sidecar 扩展 | 可并行设计，不能自动进入主线事实 | `PSE0`、`docs/sidecar/parallel_sidecar_extension_governance.md` | 主线 benchmark 或论文 claim |

## 3. 高层时间线

| 时间 | 阶段 / 任务 | 关键转折 | 当前保留方式 |
| --- | --- | --- | --- |
| 2026-03-17 | P0 | `full_qec` vs `simplified` baseline gap 被确认 | 作为物理/简化模型差异的历史起点，具体数值只按原 run 引用 |
| 2026-03-19 | P1/P2 | `static_theta_v2` 模型、量化资产和行为级自适应链路形成 | 代码与 artifact 可作为历史材料；训练可复现性以后续 `T31/T39/T40/T50` 为准 |
| 2026-03-28 | P3 software HIL | software HIL 路径打通 | 后续由 `T3/T4/T6/T12` 降级并收口为 `mock + artifact_npz + inproc` bounded path |
| 2026-03-29-31 | P3 参数/overflow 调整 | input range、overflow、默认参数经历多轮调参 | 只保留机制素材；不可把旧调参叙事写成当前 formal evidence |
| 2026-04-01 | P4 初期 | 路线从 absolute parameter regression 转向 teacher-guided residual-b | 方向仍有价值，但 formal 结果以 `T24` 后的证据层为准 |
| 2026-04-03 | 强 baseline | UKF/RLS 等 classical baseline 加入 | 作为 baseline taxonomy 保留；未来扩展需 protocol lock |
| 2026-04-04-17 | features / No TeacherParams | No TeacherParams 离线指标好，但 formal HIL 会随 seed 翻转 | 结论已降级：不能作为稳定更优主线 |
| 2026-04-27-29 | Gated v5/v8/v9 | Gated v5 一度成为强 candidate，v8/v9 边际收益低 | 作为机制和 sidecar 素材保留，不再鼓励无界超参微调 |
| 2026-05-05-08 | T0-T13 Recovery | 完成治理、边界审计、P0/P3/P4 smoke、manifest、exit review | 项目从 Recovery 进入 Controlled Development |
| 2026-05-10 | T23-T25 | P4 formal protocol lock 和 T24 frozen-set formal software revalidation 完成 | `T24` 成为历史权威 frozen-set software-HIL anchor |
| 2026-05-16-24 | T31/T36/T38/T46/T54-T58 | 训练依赖、seed 20260429、机制 trace/intervention、paper material ledger 陆续补强 | 机制结论更细，但仍不能写成完整因果闭环 |
| 2026-05-26-06-10 | T59-T70 | `statcalib` 从 smoke、isolation、fairness、FR8 benchmark、tie-break 到 closure pack | 作为 extension lane 保留；`T70` 明确 no-promotion |
| 2026-06-10 | T48/T49/T50 | `.tflite` isolated runtime、real-board gate、training/material pack 三类边界补强 | 均为 bounded evidence，不升级为部署闭环 |
| 2026-06-11 | T71/T72 | real-board gate pack 从 role-aware/regeneration 进入 provenance hardening | 当前唯一任务是 `T72`；仍无真板执行成功 |

## 4. P0-P4 当前解释

| 阶段 | 当前可保留结论 | 后续替换 / 降级说明 |
| --- | --- | --- |
| P0 | 物理基线脚本和历史 run 支持项目不是空壳；最小 smoke 入口已恢复 | 旧数值只作历史引用，不替代后续 formal evidence |
| P1 | 数据、训练、量化、导出资产存在，`static_theta_v2` 是重要历史模型族 | 训练链可迁移性、clean env、材料再生成以 `T31/T39/T40/T50` 为准 |
| P2 | 行为级仿真链路和 classical/adaptive baseline taxonomy 可复用 | 旧 P2 不能证明 P3/P4 或真板完成 |
| P3 software HIL | 当前可信边界是 bounded software HIL：`mock + model_artifact + artifact_npz + inproc` | 不能写成 true `.tflite` HIL 或 real-board HIL |
| P3 real-board | 只有 placeholder / gate / readiness / transfer-pack provenance 层证据 | 当前 host verdict 仍是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE` |
| P4 frozen benchmark | `T24` 是 frozen-set formal software-HIL anchor：四场景、五模式、`repeats=2`，`hybrid_residual_b` 均为 winner | 不等于 paper-grade expanded benchmark；不得被 `statcalib` extension lane 改写 |
| P4 extension | `statcalib` 已有 bounded extension-lane closure 和 no-promotion gate | 只可作为 extension/future-selection 素材，不能升为成熟 comparator |

## 5. 仍有效的结论

1. 项目不是空壳：`physics/`、`cnn_fpga/`、`benchmark/`、`docs/` 中有完整的历史代码、配置、artifact 和治理材料。
2. `Go` 的含义是允许继续 bounded development，不是允许无任务包扩展 benchmark、runtime、真板或论文 claim。
3. fast/slow dual-loop 与 teacher-anchored residual/control calibration 是当前最稳的方法主线。
4. `ParamMapper`、P4 runner 语义、baseline 集合、scenario matrix 和 evidence level 必须显式冻结；任何修改都应单独任务化。
5. `T24` 是当前最权威的 frozen-set software-HIL 结果锚点。
6. `T48` 只确认 current-host isolated true `.tflite` runtime，不确认默认环境或 HIL/board 集成。
7. `T49/T71/T72` 只属于 real-board gate/provenance 读侧材料，不是真板执行成功。
8. `T64`-`T70` 的 `statcalib` 是 extension lane；`T70` 的 no-promotion gate 必须随引用一起保留。
9. 机制诊断已经比早期更强，但 `T55/T56` 也削弱了简单因果叙事；论文写作必须保留 hedge。
10. `runs/` 和 `artifacts/` 是历史证据材料，不应被整体改写成新的事实来源。

## 6. 已被替换或降级的旧结论

| 旧说法 / 旧入口 | 当前处理 |
| --- | --- |
| `docs/02_experiment_plan.md` 中 2026-05-08 的“当前唯一任务：待定义” | 已由 `docs/04_task_board.md` / `docs/07_handoff.md` 的 `T72` supersede |
| `docs/follow-up_plan/README.md` 是后续计划唯一维护入口 | 已由本文档 Part II 替代；该文件只保留为退役索引 |
| P3 中出现 `real_board` mode 就代表真板 HIL 近似完成 | 降级为 placeholder/gate/readiness 证据；真板执行仍未发生 |
| `.tflite` artifact、`.tflite.json` stub、TFLite runtime、HIL runtime 可混写 | 已拆成 artifact type、stub fallback、isolated true runtime、HIL/board integration 四层 |
| No TeacherParams 离线更好，可作为主线 | formal HIL seed 翻转，不能作为稳定更优主线 |
| 继续追 Gated v10/v11/v12 超参可能是主路径 | 降级为低优先级；后续应改为机制诊断、protocol lock 或 sidecar |
| `statcalib` 可自然并入 T24 frozen table | 被 `T26/T30/T64-T70` 改写为 separate extension lane，并由 `T70` 明确 no-promotion |
| paper-ready prose 可直接推进 | 被 Research Reality Recovery Mode、claim/evidence/material ledger 和后续 result packs 约束 |
| real-board gate pack 已足够 future-host 复用 | `T71` 关闭 R30，但 `R31/T72` 指出 provenance 仍需 execution-derived / override-safe hardening |

## 7. 治理工作方式

后续所有开发继续遵守：

- 每轮只推进一个 current unique task。
- Worker 只改 Allowed files，不自动领取下一任务。
- Reviewer 默认只读，优先查 overclaim、mock/stub/placeholder、benchmark 公平性、环境省略和可复现性。
- 新任务必须有 `Allowed files`、`Forbidden scope`、`Verification`、`Docs to update`。
- 不把计划、参考建议、draft prose、sidecar output 或 historical artifact 写成完成事实。

---

# Part II：后续开发计划

## 8. 计划维护规则

1. 从 2026-06-11 起，后续计划只维护本文档 Part II。
2. `docs/follow-up_plan/README.md` 退役为索引说明，不再作为活跃计划入口。
3. 任何来自 `docs/reference/`、`docs/deep_research_reports/`、`docs/legacy_context/` 或旧 follow-up 文档的建议，必须先在这里归纳，再拆成独立任务包。
4. 计划本身不能证明结果；结果必须引用 task/review/run/artifact。
5. 当前任务状态变化时，优先同步 `docs/04_task_board.md` 和 `docs/07_handoff.md`；本文档只记录稳定路线与候选池。

## 9. 当前主线任务

当前唯一任务：

- `T72: Real-board transfer-pack provenance hardening`
- 任务包：`docs/tasks/Phase2/T72_real_board_transfer_pack_provenance_hardening.md`

当前任务边界：

- 只处理 read-only host / device / bitstream / AXI / DMA / repo-path truth 的 provenance hardening。
- 目标是让 transfer pack 的说明从固定文案变成 execution-derived / override-aware。
- 需要补 `--config`、`--mmio-path`、`--dma-path` 等 override provenance 的 focused regression。
- 不得扩展到 benchmark、HIL 语义、`.tflite`、真板成功宣称、theory branch、sidecar promotion 或 write-side MMIO/DMA/register actions。

`T72` 完成前，下面所有内容都是候选计划，不是当前任务。

## 10. 后续路线总览

后续开发按证据等级分层推进，不一次性展开全部方向：

1. 主线可信度与复现边界
2. paper-grade benchmark expansion
3. 机制诊断与 ablation
4. runtime 与 `.tflite` 补强
5. 板级语义与真板路径
6. 工程仿真补强
7. paper / 投稿路线
8. sidecar / research extension

## 11. 主线可信度与复现边界

目标：让当前已存在的主线结论更容易复查、迁移和引用。

可任务化方向：

1. 维护 `T24` frozen-set benchmark 的权威地位，不改写历史表格。
2. 将每张结果表绑定到 task、run root、config、summary helper、review。
3. 为训练材料、模型 artifact、`.tflite` artifact、runtime gate 和 real-board gate 建立更清晰的 manifest。
4. 对未来任何 benchmark rerun 先写 protocol lock，再执行。
5. 刷新 claim/evidence ledger 与 result/figure ledger，避免论文材料脱离证据来源。

验收口径：

- 新文档必须能说清“这条结果来自哪个 task 和哪个 evidence level”。
- 不把 recovery smoke、development smoke、mock-backed software-HIL、true `.tflite` runtime、real-board gate 写成同一种证据。

## 12. Paper-Grade Benchmark Expansion

这部分来自旧完成/投稿路线、深度调研报告和工程补强建议，仍有价值，但只能通过新任务包推进。

可任务化方向：

1. 强 classical baseline：fixed teacher、window variance、EKF、UKF、RLS residual、oracle-style upper/lower bound。
2. `statcalib` / prior-update baseline：只作为 extension lane 或 future-selection task，不自动进入主线冻结表。
3. learned baseline：CNN-only、teacher-guided residual-b、residual-(K,b)、compact-statistics variant。
4. scenario 扩展：random-walk drift、sinusoidal drift、burst/reset drift、unseen drift holdout。
5. 统计协议：训练 seed 与评测 seed 分离，公共随机流复用，置信区间或停止准则预先声明。
6. 指标扩展：除 `LER` 外，记录 update lag、commit/rollback、slow-loop violation、latency p50/p95/p99、overflow/saturation。

边界：

- `T45` 只锁定了 policy/protocol 分类，没有执行 broader benchmark。
- 未来 expanded benchmark 必须保留 `T24` frozen table 作为历史 anchor。
- 未经新 task 执行前，不得写成 paper-grade expanded benchmark 已完成。

## 13. 机制诊断与 Ablation

后续需要解释为什么 teacher-anchored residual/control calibration 有效，但不能让解释跑在证据前面。

可保留问题：

1. histogram 中哪些统计量最能预测 drift-induced failure：均值偏移、轴向方差、偏度、边缘峰值、时间差分、anisotropy。
2. `residual-b` 为什么在一些场景足够，在哪些 drift family 下不够。
3. teacher-only、CNN-only、residual-b、residual-K、residual-(K,b)、statcalib-only 的分层 ablation。
4. context window、histogram delta、teacher prediction、teacher params、teacher deltas 的输入通道贡献。
5. update cadence、commit cadence、rollback/fallback 与稳定性的 trade-off。

验收口径：

- 机制诊断先用小样本、frozen scenario 或 focused trace，不直接启动正式长跑。
- 诊断结论不能替代 formal benchmark，也不能把相关性写成因果证明。
- `T55/T56` 后，简单 “high committed-b is harmful” 叙事不能再无条件保留。

## 14. Runtime 与 `.tflite` 补强

当前事实：

- `T48` 已确认 current-host isolated true `.tflite` runtime 的窄路径。
- 默认环境、跨主机、部署链、HIL 链路和真板链仍未闭合。

后续方向：

1. 建立 `.tflite` runtime bootstrap，记录 Python、TensorFlow/LiteRT、artifact hash、source-vs-tflite 对照和 latency。
2. 将 `.tflite` 证据拆成 isolated current-host verification、default-env compatibility、cross-host/deployment portability、HIL/board integration 四层 gate。
3. true runtime smoke 必须显式拒绝 `.tflite.json`、`tflite_stub_service` 或 fallback predictor 通过。
4. 在 software-HIL 内引入 `.tflite` slow-loop path 前，先做最小 deterministic smoke。
5. 为 quantized runtime、fixed-point fast-loop、parameter bank 和 commit semantics 建立部署边界表。

边界：

- `T48` 不等于默认环境恢复。
- `.tflite` runtime 不等于 HIL closure。
- HIL closure 不等于 real-board validation。

## 15. 板级语义与真板路径

当前事实：

- `board_backend.py` 仍不能写成真实板级完成。
- `T49/T71` 证明的是 read-only gate / regeneration / transfer pack 边界。
- current-host verdict 仍是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`。
- `T72` 正在处理 transfer-pack provenance，而不是真板执行。

后续方向：

1. 完成 `T72`：去掉未探测即写死的 provenance 文案，让 config/path provenance execution-derived / override-aware。
2. 继续保持 host/device/bitstream/AXI/DMA/repo-path truth 的只读 gate。
3. 未来可补 board backend shadow state machine：register shadow、parameter bank shadow、commit ack/fail、DMA stale window。
4. 建立板级异常事件 taxonomy：DMA stale read、partial write、commit timeout、bank mismatch、device path unavailable。
5. 只有在设备路径、bitstream/RTL/DMA contract、地址表和权限条件满足后，才可打开真板 smoke execution task。

边界：

- 后续任何 write-side MMIO/DMA/register action 必须另有明确授权和任务包。
- `T37` 在 real-board gate/provenance 条件满足前继续 blocked。

## 16. 工程仿真补强方向

以下方向来自旧工程计划和 reference 归档材料，可作为未来候选任务池。

### 16.1 物理噪声到有效参数的离线标定层

目标：在不重写主线训练/benchmark 接口的前提下，把 `physics/noise_channels.py` 的更丰富噪声模型映射到当前有效参数 envelope。

可拆法：

- 增加离线标定脚本，回答某类物理噪声对应怎样的 `(sigma, mu_q, mu_p, theta)` envelope。
- dataset 侧新增 `source_physics_profile` / `effective_param_trace` 元数据。
- benchmark 输出区分“物理噪声来源”和“场景 envelope”。
- 先做小规模对照，不直接替换 formal benchmark。

### 16.2 Load-Aware Latency 与状态化故障模型

目标：让 slow-loop runtime 和 HIL 更接近真实运行时，而不是只使用独立抽样或独立伯努利故障。

可拆法：

- 在 benchmark 输出中增加 backlog、pending update、stale window、slow-loop queue state。
- 实现 `load-aware latency injector v1`，区分轻载、重载、拥塞。
- 将 missed update、late commit、stale parameter、fallback activation 对象化。
- 第一版只做可诊断的小样本对照，不直接替换 formal benchmark。

### 16.3 Bit-Accurate / Fixed-Point Shadow Pipeline

目标：把“接近硬件”的固定点模拟推进到更清晰的逐级位宽规范。

可拆法：

- 整理 syndrome、`K`、`b`、correction、histogram bin、accumulator 的 fixed-point 格式。
- 实现 bit-accurate shadow pipeline，只和现有 fast loop 做逐条对比。
- 用小样本 trace 找差异来源，再决定是否进入 benchmark。
- 不把 shadow pipeline 写成 RTL 等价或 FPGA 验证。

### 16.4 测量链与逻辑错误口径增强

目标：把 syndrome 输入和 LER 口径说得更清楚，提升论文解释力。

可拆法：

- 增加轻量 ADC/AFE 或 measurement readout envelope。
- 将 input range、clipping、histogram saturation 与 fast-loop 输入范围联动。
- 明确当前 LER 是哪一种有效模型口径。
- 做一组扩展型 tracker 与当前 tracker 的抽样对照。

## 17. Paper-Inspired / StatCalib / Sidecar 路线

`statcalib` 已完成一轮 extension-lane closure，并明确 `no_promotion_keep_extension_lane_only`。它可作为 future-selection、safety floor 或 calibration baseline 素材，但不能自动成为主线 comparator。

可保留设计：

1. compact histogram summary：histogram energy、centroid shift、anisotropy、short-window energy mean/std。
2. teacher confidence summary：`||delta_b||`、short-window `b` drift mean/std、teacher stability flags。
3. dual-branch / stat-calib head：histogram branch、teacher scalar branch、compact summary branch。
4. closed-loop-consistency loss：residual supervision、`b_next = teacher_b + delta_b_pred` 对齐、risk/smoothness penalty。

来自 GPT-Pro 扩展实验调研的 sidecar 候选：

1. temporal histogram stack + tiny TCN residual `b` head。
2. adaptive syndrome-only teacher + confidence-gated fallback。
3. piecewise-affine / gain-scheduled FPGA parameter bank。
4. atomic commit / rollback and transfer-boundary controller checks。

边界：

- sidecar lanes 只能在隔离分支、隔离 worktree 和 `runs/sidecar/<lane_id>/...` 下推进。
- 任何 sidecar 输出进入主线前必须通过 Captain promotion gate。
- S4/Mamba、surface-GKP、QLDPC-GKP、transformer/full decoder 等方向只保留为 research-only 或 future-work，除非后续另开任务定义。

## 18. 论文写作计划

当前更稳的论文定位：

> 本项目不是证明 CNN 全面替代经典解码器，而是证明在实时/部署约束下，保留稳定 classical teacher，并让轻量学习或校准模块修正运行时真正使用的 residual/control term，是一条更可控的自适应 GKP 解码工程路线。

应避免：

- “first neural decoder for GKP”
- “CNN 全面优于所有经典解码器”
- “完整真实 FPGA 系统已经验证”
- “`.tflite` / HIL / real-board 已经形成统一闭环”
- “statcalib 已成为成熟主线 comparator”

可保留标题方向：

1. `Runtime-Consistent Teacher-Guided Residual Decoding for Drift-Adaptive GKP Codes`
2. `Deployment-Bounded Residual Calibration for Adaptive GKP Decoding`
3. `Histogram-Conditioned Teacher-Anchored Calibration for GKP Fast-Path Decoding`

可保留贡献点：

1. dual-loop runtime-consistent GKP adaptive decoding framework。
2. teacher-anchored residual/control calibration formulation。
3. 分层证据链：frozen-set mock-backed software-HIL、extension-lane statcalib closure、isolated true `.tflite` runtime gate、real-board read-only gate/provenance boundary。

投稿前需要重新回填：

- 正式结果表与 run root。
- ablation 表和机制解释。
- `.tflite` 表，只能引用 `T48` 边界。
- real-board 表，只能引用 `T49/T71/T72` gate/provenance 边界，不能写作执行成功。

## 19. 图表与材料清单

| 图/表 | 用途 | 当前边界 |
| --- | --- | --- |
| 系统架构图 | dual-loop fast/slow path、histogram、teacher/calibration、commit | 可画概念图，但必须标注 mock/software-HIL |
| 证据等级表 | 区分 smoke、formal software-HIL、`.tflite`、real-board gate | 必须与治理文档一致 |
| 主结果表 | frozen `T24` anchor 与后续 extension lane 分开展示 | 不混表、不改写历史 |
| baseline 表 | EKF/UKF/window variance/RLS/hybrid/statcalib 等 | 是否纳入主线取决于 task |
| ablation 表 | teacher/context/features/histogram/residual-b | 未完成项标 pending |
| runtime 表 | latency、commit、rollback、overflow、saturation | 不写成真板指标 |
| real-board gate 表 | device path、bitstream、AXI/DMA、repo path truth | 当前是 NO_GO/provenance，不是执行结果 |
| claim/evidence ledger | claim -> task/review/run/artifact | 论文 reopen 前必须刷新 |
| result/figure ledger | figure/table -> script/config/run root/review | 论文 reopen 前必须刷新 |

## 20. 投稿路线

若保持 mock-backed software-HIL + 清晰 runtime boundary + paper-grade writing，当前更稳的目标是：

- IEEE Quantum Week / QCE
- IEEE Transactions on Quantum Engineering / TQE
- EPJ Quantum Technology
- Quantum Science and Technology / QST
- ACM Transactions on Quantum Computing / TQC

FCCM、ACM FPGA、DATE、ICCAD 等硬件向 venue 只有在补齐以下证据后才适合作为主目标：

1. 真实板卡或等价硬件路径。
2. 资源、时延、吞吐、接口 serialization 成本。
3. bitstream / RTL / DMA / AXI / register contract。
4. 与 software-HIL 的误差对照。

当前不能把这些 venue 写成“马上适合”的主目标。

## 21. 后续任务候选池

以下不是当前任务，只是可拆包候选。任何候选转为执行前都必须写成独立 task package。

| 优先层 | 候选任务 | 主要输出 | 验证 |
| --- | --- | --- | --- |
| Current | `T72` real-board transfer-pack provenance hardening | execution-derived / override-aware transfer pack | focused regression + docs sync |
| P1 | claim/evidence ledger refresh | claim -> evidence -> task/review 表 | 文档自查 + reviewer |
| P1 | result/figure ledger refresh | figure/table -> script/config/run root/review 表 | 文档自查 + reviewer |
| P1 | `.tflite` runtime portability audit | default env / isolated env / cross-host 差异表 | bounded runtime smoke |
| P1 | `.tflite` isolated-env bootstrap hardening | interpreter/package/artifact/source manifest | true-runtime smoke rejects stub/fallback |
| P1 | training/material reproducibility follow-up | repeated-run / cross-host / CPU-vs-GPU 边界表 | bounded train/eval smoke |
| P2 | paper-grade expanded benchmark execution protocol | scenarios、baselines、metrics、seeds、stopping | protocol review first |
| P2 | GPT-Pro extension-route triage | adopted/deferred/rejected sidecar list | docs-only protocol review |
| P2 | temporal TCN / adaptive teacher / parameter-bank sidecar design | bounded experiment spec + shared inputs | no long-run execution without new task |
| P2 | mechanism diagnosis pack | histogram/residual-b/update cadence 诊断 | focused trace / small sample |
| P2 | statcalib future-selection task | extension-lane candidate comparison | task-scoped helper + run root |
| P2 | fixed-point shadow pipeline | bit width spec + shadow diff | small trace comparison |
| P3 | board backend shadow semantics | state machine / register / DMA event model | no write-side hardware action |
| P3 | real-board smoke execution | real device smoke | only after gate conditions satisfy |
| P3 | paper draft reopen gate | outline, abstract, figures, related work | no claim upgrade without evidence |

## 22. 关键索引

目录索引：

- `docs/recovery_bootstrap/README.md`
- `docs/protocols/README.md`
- `docs/evidence_packs/README.md`
- `docs/paper_materials/README.md`
- `docs/sidecar/README.md`
- `docs/codebase_overview/README.md`
- `docs/legacy_context/README.md`

治理与当前状态：

- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

边界与 bootstrap：

- `docs/03_hil_p4_boundary_audit.md`
- `docs/recovery_bootstrap/P0_smoke_bootstrap.md`
- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`
- `docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`
- `docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md`

论文与研究材料：

- `docs/codebase_overview/README.md`
- `docs/paper_notes/README.md`
- `docs/deep_research_reports/README.md`
- `docs/paper_materials/mainline_theory_analysis.md`
- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md`
- `docs/legacy_context/reference_retired_2026-06-11/README.md`

## 23. Captain 注意事项

1. 不要把 `P3-软件 HIL` 写成 `P3-真板 HIL 已完成`。
2. 不要把 `board_backend.py` 的 placeholder 语义写成真实板级完成。
3. 不要把 `runs/`、`artifacts/` 中的历史结果改写为新的事实来源。
4. 不要无任务包启动新的 teacher-representation 长跑或正式 benchmark。
5. 不要把 `T48` 写成默认环境、HIL 或 deployment closure。
6. 不要把 `T64`-`T70` 写成 mature `statcalib` comparator promotion。
7. 不要把 `T49/T71/T72` 写成 real-board execution success。
8. 不要跳过验证就更新阶段结论类文档。
