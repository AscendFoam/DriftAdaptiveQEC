# CNN-FPGA-GKP 项目 PPT 汇报内容方案

**目标页数**：建议 12 页；如时间较短，可压缩为 10 页。  
**汇报定位**：面向项目进展汇报，既讲“为什么做、怎么推进”，也讲“已经得到哪些可量化结果、还差什么、论文怎么写”。  
**当前状态口径**：截至 2026-05-08，项目处于 `Phase 2: Controlled Development`，决策状态为 `Go`。当前唯一开发任务是 `T15: P4 multi-scenario frozen baseline bounded smoke`。

## 汇报主线

一句话主线：

> 本项目围绕漂移自适应 GKP 量子纠错，构建了一套 CNN-FPGA 双回路在线解码框架；当前最稳妥的结论不是“CNN 替代所有经典解码器”，而是“在运行时一致的工程约束下，classical teacher + lightweight CNN residual-b 修正是一条可部署、可复验、具有论文价值的路线”。

汇报要避免的表述：

- 不说 `P3 真板 HIL 已完成`，只能说 `P3 软件 HIL / mock-backed recovery path 已复验`。
- 不把 `board_backend.py` placeholder 写成真实板级完成。
- 不把 `T9/T15` 的 bounded smoke 写成完整四场景 formal benchmark 已恢复。
- 不把 `.tflite` 当前 recovery path 写成已恢复运行时证据；当前 recovery manifest 不覆盖 `.tflite` runtime。

---

## 第 1 页：标题页

**标题**：CNN-FPGA 协同的 GKP 漂移自适应量子纠错系统  
**副标题**：从物理仿真、双回路软件 HIL 到 teacher-guided residual 解码  

**页面内容**：

- 项目名：`DriftAdaptiveQEC`
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 汇报重点：
  - 项目问题与架构
  - 阶段推进过程
  - 关键实验结果
  - 当前边界与风险
  - 后续改进与论文思路

**编排建议**：

- 左侧放项目标题和一句话定位。
- 右侧放一张“双回路架构简图”：Fast loop / Slow loop / Param bank / HIL。
- 页脚标注日期：`2026-05-08`。

**讲述要点**：

- 先把项目定性为“量子纠错 + 工程系统 + 学习型自适应”的交叉项目。
- 说明本次汇报会诚实区分已完成、已复验、历史结果和待完成内容。

---

## 第 2 页：研究问题与核心挑战

**标题**：为什么 GKP 解码需要漂移自适应？

**页面内容**：

- GKP 连续变量纠错中，噪声参数会随时间漂移：
  - 噪声强度 `sigma`
  - 位移均值偏置 `mu_q / mu_p`
  - 协方差旋转角 `theta`
- 固定参数解码器在漂移场景下会逐渐失配，导致逻辑错误率 `LER` 上升。
- 工程挑战：
  - 快回路必须低时延、确定性、可硬件部署。
  - 慢回路要利用统计窗口估计漂移并更新解码参数。
  - 离线模型精度不等于在线闭环可用。

**可放数据**：

P0 物理基线显示 `full_qec` 与 `simplified` 的 LER 存在约 `0.40` 的稳定 gap：

| 场景 | full_qec LER | simplified LER | gap |
| --- | ---: | ---: | ---: |
| `linear_low` | 0.4237 | 0.0205 | 0.4032 |
| `step_mid` | 0.42665 | 0.01855 | 0.40810 |
| `sinusoidal` | 0.42530 | 0.02405 | 0.40125 |
| `random_walk` | 0.41200 | 0.01565 | 0.39635 |

**编排建议**：

- 左侧讲物理问题：漂移导致固定解码失效。
- 右侧放 P0 gap 小表或柱状图。

**讲述要点**：

- P0 的意义不是证明某个模型强，而是提醒后续工程验证不能依赖过度简化的物理口径。

---

## 第 3 页：系统方案：CNN-FPGA 双回路架构

**标题**：快回路确定性解码，慢回路学习型自适应

**页面内容**：

- Fast loop（FPGA 侧）：
  - 每周期执行线性解码：`Delta = K @ s + b`
  - 累积 `32 x 32` syndrome histogram
  - 关注 fixed-point、cycle budget、overflow
- Slow loop（ARM/CNN 侧）：
  - 基于窗口级 histogram 估计漂移
  - 通过 teacher + CNN residual 修正更新 `(K, b)`
  - 通过 param bank stage / commit 切换参数
- HIL / benchmark：
  - 软件 HIL 复验双回路调度
  - P4 benchmark 在 HIL wrapper 上运行

**编排建议**：

- 全页放流程图，按时间尺度分上下两层：
  - 上层：5 us 级 fast loop
  - 下层：10-100 ms 级 slow loop
- 箭头标出 histogram window、model inference、param commit。

**讲述要点**：

- 强调 CNN 不是直接替代快回路，而是在慢回路里生成更好的参数更新。
- 这也是论文定位从“模型精度”转向“部署一致系统”的关键。

---

## 第 4 页：项目推进路线：P0 到 P4

**标题**：从物理可信度到工程闭环的分阶段推进

**页面内容**：

| 阶段 | 目标 | 当前结论 |
| --- | --- | --- |
| P0 | 物理基线与简化模型差异 | 已确认 full/simplified gap |
| P1 | Tiny-CNN 静态参数回归 | 已通过，float/int8 精度接近 |
| P2 | 行为仿真与公平基线 | 自适应闭环优于固定/延迟基线 |
| P3 | 软件 HIL / 部署链路 | mock-backed software HIL 已复验 |
| P4 | 多场景 benchmark 与机制分析 | 历史 formal 结论成立，Phase 2 正在补 bounded evidence |

**关键状态**：

- Phase 0 / Phase 1 recovery 已完成 T1-T13。
- Phase 2 进入受控开发，当前任务是 `T15`。
- 当前开发策略：先固化 P4 benchmark 证据，再补训练链、`.tflite`、真板 readiness manifest。

**编排建议**：

- 用横向时间轴展示 P0-P4。
- 每个阶段只放一句“已拿到什么事实”。

**讲述要点**：

- 项目不是一次性跳到 benchmark，而是先做物理口径、模型、闭环、HIL，再做强 baseline。

---

## 第 5 页：P1 成果：CNN 感知器件已通过精度与量化验收

**标题**：Tiny-CNN 能可靠识别静态漂移参数

**页面内容**：

P1 修正点：

- 数据从各向同性高斯改为各向异性高斯，使 `theta_deg` 可辨识。
- 对 `theta_deg` 提高损失权重，避免被更容易的标签淹没。

P1 核心指标：

| 模型 | MSE | MAE | R2_mean |
| --- | ---: | ---: | ---: |
| float | 0.293336 | 0.220503 | 0.994352 |
| int8 | 0.297742 | 0.221944 | 0.994212 |

逐标签 float R2：

- `sigma = 0.997613`
- `mu_q = 0.996473`
- `mu_p = 0.998459`
- `theta_deg = 0.984862`

**编排建议**：

- 左侧放“问题 -> 修正”的两步箭头。
- 右侧放 float / int8 指标表。

**讲述要点**：

- P1 的价值是建立慢回路可用的轻量感知器件。
- int8 几乎不退化，为后续部署一致性提供基础。

---

## 第 6 页：P2 成果：运行时闭环优于公平基线

**标题**：从离线回归转向运行时一致闭环

**页面内容**：

P2 关键修正：

- 去掉“作弊型 mock”基线，改用：
  - `fixed_baseline`
  - `oracle_delayed`
- 修正 `ParamMapper`：
  - `K = C(C+R)^-1`
  - `b = (I-K)mu`

P2 核心结果：

| 场景 | fixed_baseline | oracle_delayed | model_artifact | int8_artifact |
| --- | ---: | ---: | ---: | ---: |
| `linear_med` | 0.816067 | 0.764239 | 0.696206 | 0.701372 |
| `step_large` | 0.937822 | 0.926650 | 0.731906 | 0.732072 |
| `sinusoidal_mid` | 1.019033 | 1.020556 | 0.759289 | 0.761878 |

工程指标：

- `commit_count_mean = 7.0`
- `fast_cycle_violation_rate_mean = 0.0`
- `slow_update_violation_rate_mean = 0.0`

**编排建议**：

- 用三组柱状图展示 LER 下降。
- 页脚放“float 与 int8 基本重合”。

**讲述要点**：

- P2 证明改进来自运行时参数控制，而不是不公平 baseline 或量化偶然性。

---

## 第 7 页：P3 成果与边界：软件 HIL 已复验，真板仍未完成

**标题**：HIL 链路已可复验，但仍是 software/mock-backed 口径

**页面内容**：

当前 recovery 复验路径：

- backend：`mock`
- slow-loop mode：`model_artifact`
- inference backend：`artifact_npz`
- inference service：`inproc`
- artifact：`static_theta_v2` `.npz`

T12 确定性复验结果：

- 两次 run 的 `hil_summary.json` SHA256 一致。
- 两次 run 的 `hil_events.json` SHA256 一致。
- `n_windows_ready = 2`
- `n_slow_updates_finished = 2`
- `n_commits_applied = 2`
- `final_ler = 0.454375`
- `overflow_rate = 0.002`

必须明确的边界：

- 不是 `real_board`。
- 不是 `.tflite` runtime 复验。
- 不能写成“真板 HIL 已完成”。

**编排建议**：

- 左侧放一张“已复验路径”链路图。
- 右侧放一块醒目的“边界说明”。

**讲述要点**：

- 这是项目可信度恢复最关键的一页：讲清楚“能复验什么”和“不能宣称什么”。

---

## 第 8 页：P4 主结果：Hybrid Residual-B 是当前正式主线

**标题**：最强主线不是绝对参数回归，而是 teacher + residual-b

**页面内容**：

历史正式强 baseline 结果：

| 方法 | 平均 LER |
| --- | ---: |
| `Hybrid Residual-B` | 0.798332 |
| `UKF` | 0.817974 |
| `Constant Residual-Mu` | 0.825719 |
| `RLS Residual-B` | 0.827908 |
| `EKF` | 0.828369 |

核心解释：

- 修正后的 `UKF` 是当前最强经典 baseline。
- `Hybrid Residual-B` 仍领先 `UKF`，平均 LER gap 约 `0.019642`。
- 优势不是靠更激进控制换来的：
  - `correction_saturation_rate = 0`
  - `aggressive_param_rate = 0`

当前 Phase 2 口径：

- 这组是历史正式结论，应作为论文主线参考。
- 当前正在通过 `T15/T16` 补更强的 bounded development evidence。

**编排建议**：

- 用排序柱状图显示五种方法 LER。
- 加一句中心结论：`Hybrid Residual-B > UKF`，但不夸大成全面替代经典解码器。

**讲述要点**：

- 这一页是项目成果的“论文价值锚点”。
- 注意说明历史 formal 结果与当前 recovery/development 证据的层级区别。

---

## 第 9 页：机制分析：为什么是 teacher-guided residual？

**标题**：学习模块最有价值的角色是修正 teacher 的运行时偏差

**页面内容**：

已经形成的机制判断：

- 直接回归绝对物理参数不一定最适合在线闭环。
- 离线监督指标改善不等于 formal HIL 改善。
- `teacher params` 的问题更像表征设计问题，而不是简单保留/删除。
- 当前主线：
  - 保留稳定 classical teacher。
  - CNN 只预测对快回路控制偏置 `b` 有用的 residual。

teacher-representation 进展：

- `Gated v5` 只保留 4 个 teacher 标量：
  - `teacher_b_q`
  - `teacher_b_p`
  - `teacher_delta_b_q`
  - `teacher_delta_b_p`
- 分块 paired 复验中：
  - `Full = 0.758829`
  - `Gated v5 = 0.618195`
  - gap = `-0.140634`
  - `Gated v5` 在 `12` 个 seed-scenario 对照中赢 `9` 个

谨慎表述：

- `Gated v5` 是当前最强 teacher-representation 候选。
- 但尚未完全替代 `Full / Hybrid Residual-B` 正式主线，因为 `seed=20260429` 下收益收缩。

**编排建议**：

- 用“Full teacher broadcast -> compact scalar gated branch”的对比图。
- 右下角放 Gated v5 的 4 个标量。

**讲述要点**：

- 这一页连接成果与后续论文思路：核心创新不是 CNN 大，而是信息表征与闭环语义更匹配。

---

## 第 10 页：当前 Phase 2 状态与 T15 任务

**标题**：从 recovery smoke 走向 bounded benchmark evidence

**页面内容**：

当前阶段：

- `Phase 2: Controlled Development`
- 决策状态：`Go`
- 当前唯一任务：`T15`

T15 运行范围：

- Config：`cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Scenarios：
  - `static_bias_theta`
  - `linear_ramp`
- Modes：
  - `ekf`
  - `ukf`
  - `constant_residual_mu`
  - `rls_residual_b`
  - `hybrid_residual_b`
- Repeats：`2`
- Seed policy：`--paired-seeds`

T15 不是：

- 不是完整四场景 formal benchmark。
- 不是 `real_board`。
- 不是 `.tflite` runtime evidence。

**编排建议**：

- 左侧放 T9 -> T14 -> T15 -> T16 的链路。
- 右侧放 T15 bounded matrix。

**讲述要点**：

- 说明当前项目已经从“修复可信度”进入“补强证据”，但仍控制范围，避免长跑失控或口径混乱。

---

## 第 11 页：后续改进路线

**标题**：补证据、补 manifest、补边界，而不是盲目扩线

**页面内容**：

近期优先级：

1. `T15/T16`：P4 bounded evidence 与 gate review
2. `T17`：训练链独立 manifest / bootstrap
3. `T18`：`.tflite` export/runtime manifest 与 smoke plan
4. `T19`：tracked cache cleanup manifest
5. `T20`：real-board HIL readiness checklist

研究方向：

- `seed=20260429` 失败/收益收缩机制诊断。
- `Gated v5` 后续小改版：
  - 弱化 delta 项
  - 保留 `teacher_b_q / teacher_b_p`
  - 控制 residual scale / clip
- `paper_inspired_statcalib_v1`：
  - compact histogram summary
  - teacher stability summary
  - light stat-calib branch
  - closed-loop consistency loss

工程补强方向：

- load-aware latency injector
- stateful fault injector
- bit-accurate control pipeline
- real-board readiness，但不提前宣称完成

**编排建议**：

- 分三栏：证据补强、研究改进、工程补强。
- 每栏只放 3-5 个关键词。

**讲述要点**：

- 后续不是“再调一个模型”，而是把 benchmark、环境、部署和机制解释逐步收口。

---

## 第 12 页：论文思路与总结

**标题**：论文主线：运行时一致的 teacher-guided residual adaptive decoding

**页面内容**：

推荐论文定位：

- 工程系统型：双回路 CNN-FPGA 自适应解码框架。
- 方法机制型：teacher-guided residual-b 在线修正。
- 部署一致型：runtime-consistent data / HIL / artifact / param bank。

一句话论文主张：

> 在部署语义一致的双回路实时解码框架下，保留经典 teacher 并让轻量 CNN 学习 residual-b 修正，比直接做绝对参数回归更有效，也比当前最强经典自适应 baseline `UKF` 更优。

建议投稿方向：

- 稳妥目标：`QCE` / `TQE` / `EPJ Quantum Technology`
- 补强后可考虑：`QST` / `npj Quantum Information` / `ACM TQC`
- 硬件显著补强后再考虑：`FCCM` / `ACM FPGA` / `ICCAD` / `DATE`

最后总结：

- 已有成果：P0-P2 基线和闭环已验证，P3 software HIL recovery path 可复验，P4 历史 formal 主线结果有论文价值。
- 当前边界：真板 HIL、`.tflite` current recovery、完整 formal benchmark 仍需补证据。
- 下一步：用 T15/T16 把 P4 evidence 固化，再补训练链、TFLite 和真板 readiness。

**编排建议**：

- 上半页放论文定位三角：System / Method / Deployment。
- 下半页放三句话总结：已完成、当前边界、下一步。

**讲述要点**：

- 用“有价值但不夸大”的姿态收尾。
- 强调项目创新性在于工程约束下的可部署自适应解码框架，而不是单纯神经网络刷分。

---

## 10 页压缩方案

如果汇报时间只有 10 分钟或页数必须控制在 10 页，可合并：

1. 合并第 5 页和第 6 页：作为“P1/P2：模型与闭环验收”。
2. 合并第 10 页和第 11 页：作为“Phase 2 当前任务与后续路线”。

压缩后结构：

1. 标题
2. 研究问题
3. 双回路架构
4. P0-P4 推进路线
5. P1/P2 模型与闭环结果
6. P3 软件 HIL 结果与边界
7. P4 主结果
8. teacher-guided residual 机制
9. Phase 2 当前任务与后续路线
10. 论文思路与总结

---

## 建议准备的图表清单

1. 双回路架构图：Fast loop / Slow loop / Param bank / HIL。
2. P0 full_qec vs simplified LER gap 柱状图。
3. P1 float/int8 R2 指标表。
4. P2 四模式 LER 对比柱状图。
5. P3 software HIL recovery path 链路图。
6. P4 strong baseline LER 排序图。
7. teacher representation：Full broadcast vs Gated scalar branch 对比图。
8. Phase 2 task roadmap：T14 -> T15 -> T16 -> T17/T18/T19/T20。

---

## 数据与口径来源

- `docs/02_experiment_plan_simplified.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/P4_benchmark_development_protocol.md`
- `docs/CNN_FPGA_GKP_阶段结论.md`
- `docs/CNN_FPGA_GKP_paper_inspired分支实验设计草案.md`
- `docs/CNN_FPGA_GKP_项目完成目标与投稿路线报告.md`
- `docs/CNN_FPGA_GKP_论文提纲_摘要_贡献点草稿.md`
