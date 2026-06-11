# Deep Research Reports

本目录保存深度研究、查重和论文定位材料。原报告保留为历史研究输入；本 README 只作为截至 2026-06-11 的状态索引，用来标记哪些建议已经被后续任务覆盖，哪些内容已经过时、需要降级解读或重新任务化。

当前项目事实仍以 `docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 和对应 task/review 文档为准。本目录中的报告不能单独作为“当前已完成”或“硬件已验证”的证据。

## 当前权威边界

- 当前阶段是 `Phase 2: Controlled Development`，决策状态为 `Go`。
- 当前唯一任务是 `T72: Real-board transfer-pack provenance hardening`；它是迁移包溯源与回归加固任务，不是 benchmark、`.tflite`、真板联调或论文结论任务。
- `T24` 已形成冻结集 formal software revalidation，但边界是 mock-backed software-HIL；不能写成 `.tflite`、真板或 paper-grade expanded benchmark。
- `T45` 已对深度研究提出的 broader benchmark 进行协议分类和锁定；未被采纳的扩展项仍需单独任务化。
- `T48` 已确认 preserved `static_theta_v2` 路径的 isolated current-host true `.tflite` runtime，但不等于默认环境、跨主机部署或 HIL/真板闭环恢复。
- `T49` 的 current-host real-board gate 结论是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`；`T71`/`T72` 只围绕 gate pack、迁移包和 provenance 加固，不能写成真板验证完成。
- `T64`-`T70` 的 `statcalib` 结果是有界 extension lane 结论；不能提升为主线成熟 comparator，也不能覆盖冻结的 T24 表格。

## 文件状态

| 文件 | 用途 | 当前判定 |
| --- | --- | --- |
| `TinyCNN辅助GKP纠错.md` | 早期 Tiny-CNN/GKP 漂移适应、PSSI、RFSoC/TFLite Micro/HIL 架构设想 | 历史可行性报告；部分思路已被当前工程吸收，硬件和部署叙事需要降级解读 |
| `进一步的深度研究结果.md` | 相关工作、撞车风险、论文定位、benchmark/机制/runtime/真板优先级建议 | 仍有参考价值；部分建议已被任务覆盖，部分已被治理文档改写为 deferred 或 boundary-only |
| `GPT-Pro的调研分析.md` | 对 `CNN_FPGA_GKP_theory_note_draft.tex` 的结构化审阅、论文定位、claim 边界和相关工作撞车风险 | 已从 `docs/reference/` 迁入；可作为论文写作和 novelty-risk 输入，不作为当前完成态证据 |
| `GPT-Pro有关扩展实验的建议.md` | 受控并行扩展路线、sidecar experiment lanes、TCN/adaptive teacher/piecewise-affine bank 等候选分析 | 已从 `docs/reference/` 迁入；可作为后续任务候选池输入，所有实验必须重新任务化 |
| `DriftAdaptiveQEC 查重与相关工作深度调研报告 (1).docx` | 查重与相关工作深度调研原始文档 | 与 `进一步的深度研究结果.md` 主题高度重叠；保留为原始调研件，引用前应先转写/核对 |

## 后续可用于项目开发的内容

### `TinyCNN辅助GKP纠错.md`

可继续用于后续开发的内容主要是早期架构设想和任务素材，而不是当前事实结论：

- PSSI / syndrome histogram-as-image 表征思路，可继续作为 slow-loop 特征工程、可视化诊断和 ablation 任务的来源。
- 漂移类型归纳，包括均值偏移、方差膨胀、相关噪声、旋转/相位漂移等，可转化为后续 synthetic drift family、holdout scenario 或机制诊断任务。
- 轻量 CNN / Micro-CNN / MobileNetV3 对比，可用于未来模型压缩、量化推理、CPU slow-loop runtime 或 `.tflite` 后续任务的候选设计。
- 双回路架构，即 fast loop 保持简单线性/查表路径，slow loop 低频更新参数，可用于继续细化 parameter bank、atomic commit、rollback 和 fallback 语义。
- 32x32 直方图压缩、FPGA 端累积、AXI/寄存器交互等硬件协同想法，可作为未来 real-board transfer、HDL feasibility 或 board smoke 前置需求清单，但不能写成当前已实现。
- baseline 草案，包括静态参数、滑窗统计、EKF、Tiny-CNN 等，可作为后续 bounded benchmark task 的候选池；是否纳入正式协议仍需单独锁定。

### `进一步的深度研究结果.md`

可继续用于后续开发的内容更偏论文路线、benchmark 设计和风险控制：

- 论文主张收窄建议仍然有用：后续应围绕 histogram-conditioned、teacher-anchored、deployment-bounded residual calibration 组织贡献，而不是宽泛 neural GKP decoder。
- 相关工作和撞车风险列表可作为 paper related work、introduction risk framing 和 reviewer 预案的初稿来源。
- benchmark 设计建议可作为未来 paper-grade expanded benchmark 的输入，包括 strong classical baselines、statcalib/prior-update baseline、learned baselines、漂移家族、训练/评测随机性隔离、置信区间和 latency/commit/rollback 指标。
- 机制诊断问题可继续任务化：histogram 到底保留了哪些 drift/failure 相关统计量、residual-b 为什么足够或何时不够、update cadence 与稳定性的 trade-off。
- `statcalib` 分支定位仍可复用为 safety floor / calibration baseline 的论证来源，但当前只能沿 extension lane 边界继续推进。
- runtime 优先级建议仍有工程价值：在真板之前优先补强 true `.tflite`、量化软件路径、fixed-point 和 deployment boundary 证据。
- roadmap 顺序可作为后续 captain 拆任务参考：先 benchmark/protocol，再机制，再 runtime，最后 real-board credibility booster。

### `GPT-Pro的调研分析.md`

可继续用于后续开发的内容主要是论文定位、claim 收窄和审稿风险预案：

- 论文主张应限定为 two-timescale adaptive affine decoder / deployment-constrained residual calibration，而不是宽泛的 neural QEC decoder、adaptive QEC decoder 或 FPGA decoder 首创。
- 证据边界表述可直接作为写作自查清单：当前可写 mock-backed software-HIL、frozen-set revalidation 和 current-host isolated `.tflite` gate；不能写真实 FPGA board、完整 embedded runtime 或 paper-grade expanded benchmark 已完成。
- 理论部分可复用 local linear-MMSE / affine approximation 的降级口径，并补充 modular lattice boundary、多峰 posterior、符号约定和 notation table 的一致性检查。
- related work 分组很有价值：GKP analog/soft information、adaptive priors、calibration-conditioned neural decoders、real-time/FPGA QEC decoders 都应进入后续论文的撞车风险地图。
- benchmark 建议可转写为未来 protocol：从 parameter regression 转向 logical error、regret-to-oracle、adaptation lag、latency/resource/commit/fallback 这类 closed-loop 与 deployment-contract 指标。
- baseline 清单可作为后续 paper-grade benchmark 输入，包括 closest-integer / nearest-lattice、known-noise Bayes/ML、static calibrated affine、periodic oracle recalibration、teacher-only、FiLM/direct CNN/residual variants 和 fixed-point variants。
- 文中外部文献链接进入正式论文前必须重新核对来源、版本、DOI/citation key 和引用上下文，不能直接复制“to our knowledge”或首创句。

### `GPT-Pro有关扩展实验的建议.md`

可继续用于后续开发的内容主要是受控 sidecar 扩展路线，而不是主线改写：

- 推荐的总体策略仍可采用：保留快慢双回路与 FPGA fast-loop contract，把扩展路线定义为 sidecar lanes，不改写 frozen baseline table、不替代 T24/T64-T70 结论。
- 最适合进入后续候选池的三条路线是 temporal histogram stack + tiny TCN residual `b` head、adaptive syndrome-only teacher + confidence-gated fallback、piecewise-affine / gain-scheduled FPGA parameter bank + atomic commit / rollback。
- 共享协议思路可复用：syndrome/histogram 缓存、teacher 输出、`b` residual target、fixed-point envelope、bank-switch safety check、paired seeds 和小规模 bounded replay。
- research-only 路线应保留但降级：S4/Mamba 只适合 toy simulation，surface-GKP / QLDPC-GKP 主要用于 paper positioning 或 future-work，不适合当前 Phase 2 短期工程实验。
- 不建议当前开启的路线也应保留为负面清单：大型 transformer/full decoder、raw time-series 全量输入、diffusion/autoregressive decoder、把 real-board HIL 或 true `.tflite` recovery 当作普通扩展实验。
- real-board HIL 和 true `.tflite` recovery 应被视为 gate/integration milestone，不是可随主线并跑的算法 extension lane。
- 若未来采用该报告中的任何实验，必须先写成单独任务包，明确 Allowed files、Forbidden scope、Verification，并声明不会提升为主线 comparator 或硬件完成态。

### `DriftAdaptiveQEC 查重与相关工作深度调研报告 (1).docx`

可继续用于后续开发的内容主要是调研原始材料和论文写作素材：

- 可作为 related work 索引源，帮助追踪 GKP soft information、hardware-conditioned neural decoder、real-time FPGA QEC、adaptive/drift-aware control 等相邻领域。
- 可提取 executive summary 和 collision-risk 叙事，作为未来论文定位、贡献边界和 novelty risk review 的初稿。
- 可用于补齐参考文献候选清单，但进入正式文档前需要重新核对来源、补全 citation key，并避免复制未验证的宽泛首创表述。
- 可作为 deep research 到开发任务的溯源材料；真正进入任务板前仍需转写为明确的 Allowed files、Forbidden scope 和 Verification。

## 已完成或已有当前状态的内容

- 直方图/窗口统计、TinyCNN/CNN slow-loop 方向已经进入当前代码和材料体系，但当前可信边界仍是受控开发与软件验证，不是真板闭环。
- P4 frozen-set formal software revalidation 已完成为 mock-backed software-HIL 证据；它只支撑冻结集软件结论。
- 机制诊断、FR6/FR7 类材料已经有后续任务覆盖一部分，但仍不能替代更强的因果机制证明。
- `statcalib` 已作为 extension lane 完成一轮受控评估和收口；当前结论强调不提升、不改写主线表格。
- true `.tflite` runtime 已在 current host 的 isolated path 上确认一条 preserved `static_theta_v2` 路径；默认环境、跨主机、部署链和真板链仍未闭合。
- real-board 方向当前完成的是 gate/pack/provenance 层面的材料化与加固；真实设备路径仍是 `NO_GO`，没有 hardware-validated 结果。

## 过时或不匹配的读法

- 不应继续使用“first neural decoder for GKP”“first adaptive QEC under drift”“first hardware-aware neural QEC decoder”这类宽泛首创叙事；后续论文主张应收窄到 histogram-conditioned、teacher-anchored、deployment-bounded 的组合贡献。
- `TinyCNN辅助GKP纠错.md` 中 RFSoC、TFLite Micro、FPGA 直方图累积、硬件在环验证等内容是架构设想，不是当前仓库已经验证的事实。
- 深度研究报告中建议的 broadened benchmark matrix、更多 drift family、更多 seeds、置信区间和强 classical baselines，并未全部成为当前已完成 benchmark；未采纳部分需要独立任务包。
- 不能把 T24 的 mock-backed software-HIL 结果写成 `.tflite`、real-board HIL 或 paper-grade expanded benchmark。
- 不能把 T48 的 isolated true `.tflite` runtime 写成默认环境恢复、完整部署闭环或真板运行。
- 不能把 T49/T71/T72 的 gate pack、migration pack 或 provenance hardening 写成真板验证通过。
- 不能把 `statcalib` extension lane 写成主线成熟 comparator、唯一最优方案或冻结表格替代。

## 后续使用规则

1. 将本目录报告作为 idea/literature/source material 使用，不作为当前完成态证据。
2. 从报告中搬运任何 claim 到主线文档前，先交叉检查 `docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 和对应 task/review 文档。
3. 若要恢复或推进报告中的硬件、benchmark、`.tflite`、statcalib 或论文包装建议，必须先写成有 Allowed files、Forbidden scope、Verification 的独立任务包。
4. 更新阶段结论类文档前，必须有对应验证输出；不得仅凭本目录报告改写当前项目事实。
