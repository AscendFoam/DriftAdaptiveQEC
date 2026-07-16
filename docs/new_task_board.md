# CNN-FPGA GKP 项目任务板

来源文档：

- `docs/rough_plan.md`
- `docs/experiment_plan.md`

用途：本任务板用于跟踪项目从问题冻结、文献地图、理论建模、实验式数字孪生、多保真仿真、三时间尺度控制、证据门、低成本 FPGA 实测与 HIL，到论文和可选真实 GKP 数据/硬件接入的全过程。所有未执行任务初始状态均为 `Todo`。每完成或推进一个任务，应同步更新 `状态`、`负责人/分支`、`产物`、`验证` 和 `备注`。

状态枚举：`Todo`、`In Progress`、`Blocked`、`Review`、`Done`、`Dropped`。

当前推荐任务：`T5.1.1`。

## 任务记录规则

每个 task 至少保留以下记录：

| 字段 | 必填内容 |
| --- | --- |
| 输入 | 使用的数据、公式、代码模块或文档 |
| 输出 | 图、表、测试、模块或论文文字 |
| 通过标准 | 数值误差、baseline 对齐、覆盖率或审阅条件 |
| 失败分支 | 若不通过，下一步应修改或转向什么 |
| 记录 | lab notebook 条目、结果产物或 review note |

单 task 粒度规则：一个 task 只回答一个科学问题、只修改一个模块、只生成一张核心图或一个验证表。不要把“训练 CNN + 改仿真器 + 调 baseline + 写论文”混成一个 task。

## v2.2 补强边界与迁移说明

本任务板于 2026-07-12 根据三份 GKP 实验/综述材料完成 v2 重构，于 2026-07-13 根据 PRL `Non-Markovian feedback for optimized quantum error correction` 完成 v2.1 定向补强，并在同日根据六篇补充论文的标题、摘要、实验/数值结果、结论和关键结果图完成 v2.2 小幅补强。执行边界如下：

- 原 Phase 2—5 中未执行且仍适用的 task ID 尽量保留，但通过标准升级为 sBs-first、事件状态感知和证据门驱动。
- 原 v1 Phase 6“论文初稿先行”与 Phase 7“真实 FPGA 实验补充”交换顺序：低成本 FPGA 实测和 HIL 必须先于主图/论文冻结。
- 原 v1 `T6.*`—`T7.*` 均未开始，现按 v2 新顺序重新编号；没有完成证据被删除。
- 真实 GKP 数据、微波谐振腔、transmon 或量子控制链路接入统一降为可选 Phase 8，不阻塞第一篇论文。
- v2.1 增加 Feedback-GRAPE teacher、memory-specific baseline、有限时域 control oracle 和 teacher-to-student 蒸馏；若可行性门失败，现有 drift/regime-aware MAP-LUT 主线继续成立。
- v2.2 只增加 noise-transfer 中保真度代理、QEC-matrix/Petz channel-recovery bound 和 top-K lattice-coset MAP；不重排阶段，不把 surface-GKP、多模 GKP、Knill/P-Steane 或物理 squeezing 扩成主线。
- `docs/rough_plan.md` 保持冻结；三次实质变更分别在 `docs/experiment_plan.md` 第 14、15、16 节登记。

## 暂定论文 claim contract

在 `T1.4.1` 正式冻结前，所有任务遵守以下板级边界：

| 层级 | 允许表述 | 禁止表述 |
| --- | --- | --- |
| 软件仿真 | experiment-informed / protocol-aligned / simulation-derived | 真实量子实验、真实 beyond-break-even |
| 综合与定点模型 | hardware-aware、fixed-point/synthesis estimate | 真实 FPGA latency/resource 已测 |
| 约 300 元 FPGA | measured digital control-plane latency/resource、HIL/replay | 真实微波生成、真实量子读出、真实 cavity/transmon 控制 |
| 可选真实硬件 | 仅在未来确有数据和接入证据后表述 | 用计划或模拟替代真实实验 |

核心贡献暂定为：实验协议启发的 syndrome-history-aware GKP 经典控制架构，由主机估计连续漂移和离散健康状态，由 model-aware recurrent teacher 发现短时 history-dependent sBs 控制规律，再蒸馏为低成本 FPGA 可执行的确定性定点 student；若 teacher/distillation 证否，则回退到 run-length-aware MAP-LUT。证据来自多保真数字孪生、强 baseline、因果故障注入、真实板卡测量和 HIL。

---

## Phase 0：问题冻结与文献地图

### Milestone 0.1：冻结研究对象

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T0.1.1 | Done | 将主对象冻结为 repeated quantum-memory error correction 下的 single-mode square approximate GKP qubit。 | 完成 `docs/tasks/T0.1.1_scope_freeze.md`；明确排除外层码、多模格码作为主线，以及 GKP 态制备优化作为主线。 | `experiment_plan.md:57` |
| T0.1.2 | Done | 冻结平台抽象，优先采用 superconducting cavity + transmon-like syndrome extraction，同时保持数学模型平台无关。 | 完成 `docs/tasks/T0.1.2_platform_abstraction.md`；包含 cycle diagram：idle noise -> syndrome extraction -> ADC/measurement -> decoder -> frame/correction -> next cycle。 | `experiment_plan.md:86` |
| T0.1.3 | Done | 冻结 CNN 慢回路与 FPGA 快回路的双回路接口。 | 完成 `docs/tasks/T0.1.3_dual_loop_interface.md`；接口表覆盖输入、输出、更新频率、位宽和 fallback policy。 | `experiment_plan.md:92` |

### Milestone 0.2：文献矩阵

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T0.2.1 | Done | 建立 Zotero/文献表，覆盖 GKP 基础/有限能量/实验、GKP 本体解码、自适应噪声估计、NN/QEC/FPGA 实时解码四条线。 | 完成 `docs/literature_matrix.md` 和 `docs/tasks/T0.2.1_literature_matrix.md`；40 篇均记录噪声模型、解码器/方法、指标、finite-energy、hardware real-time、证据等级与 Zotero 覆盖；机械检查无空字段、无 DOI/arXiv 重复。 | `experiment_plan.md:130` |
| T0.2.2 | Done | 写出 gap statement。 | 完成 `docs/gap_statement.md` 和 `docs/tasks/T0.2.2_gap_statement.md`；中文主稿、五段英文 Introduction、7 个可证伪 RQ、direct baseline/oracle/bound contract 和 hardware evidence ladder 完整；30 个 citation key 校验通过。 | `experiment_plan.md:141` |

---

## Phase 1：理论模型与指标体系

### Milestone 1.1：理想 GKP syndrome-level 解码模型

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T1.1.1 | Done | 推导单个 quadrature 的 standard binning 及高斯随机位移下的逻辑翻转概率。 | 完成 `physics/ideal_gkp_decoder.py`、`tests/test_ideal_gkp_decoder.py` 和 `docs/tasks/T1.1.1_standard_binning.md`；12 tests 覆盖 alias parity、边界、跨模块 convention、两种解析式和 40 万样本 Monte Carlo。 | `experiment_plan.md:157` |
| T1.1.2 | Done | 基于偶/奇逻辑陪集似然和 LLR 推导 MAP / soft-output likelihood。 | 完成 `coset_likelihood_1d`、`llr_1d`、`map_decode_1d` 和三模式入口；focused `24 passed`，覆盖独立 alias 穷举、period normalization、mean 跨 cell、prior、rare-event 下溢和失败分支。 | `experiment_plan.md:174` |
| T1.1.3 | Done | 将解码扩展到二维相关 `(q,p)` 噪声。 | 完成 covariance、四陪集 likelihood、joint/independent MAP；focused `34 passed`，`rho=0` 因子化通过；强相关配对 MC joint error `0.23067` vs independent `0.30883`，`z=29.60`。 | `experiment_plan.md:208` |

### Milestone 1.2：近似 GKP 有限能量模型

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T1.2.1 | Done | 定义 approximate GKP 态族，至少支持 Gaussian peaks + Gaussian envelope 和 damped/finite-energy projector model。 | 完成 normalized 两态族、四逻辑态、wavefunction/syndrome/sampled Wigner；focused `14 passed`，解析范数、独立 Mehler、截断收敛与 Wigner marginal/negative-volume 均验证。 | `experiment_plan.md:236` |
| T1.2.2 | Done | 通过逻辑通道定义 finite-energy-aware decoding，而不是只看是否跨越半格边界。 | 完成 arbitrary parity decoder -> Pauli channel、PTM/`F_e/F_avg`、finite-state/noise alias response；focused `13 passed`，12 万样本 MC 与解析误差相差 `1.74 SE`。 | `experiment_plan.md:247` |
| T1.2.3 | Done | 复现至少一条 finite-energy 或 memory-assisted approximate GKP decoding 的文献趋势。 | 完成 train/eval 分离的五点 shrinkage sweep；focused `9 passed`，`Delta=0.60->0.22` 时 fitted gain `0.363->0.801`、logical advantage `0.02406->0`，前三点 paired z 均大于 16。 | `experiment_plan.md:266` |

### Milestone 1.3：非平稳噪声漂移模型

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T1.3.1 | Done | 定义 mean drift、variance drift、loss drift、outlier-rate drift、step drift、telegraph drift 和 burst drift。 | 完成 `physics/drift_processes.py`、17 项 direct tests 和 `docs/tasks/T1.3.1_drift_process_generator.md`；七类 process、full `DriftState`、mixture sampler、Markov/burst persistence、固定 seed 前缀和 legacy `run_with_drift` adapter 均验证；全 tests/ `153 passed,4 failed`（R-N012）。 | `experiment_plan.md:282` |
| T1.3.2 | Done | 定义知道真实时间依赖噪声状态 `theta_t` 的 oracle MAP。 | 完成 `physics/oracle_map.py`、9 项 direct tests 和 `docs/tasks/T1.3.2_oracle_map.md`；full-state mean/covariance/outlier-mixture likelihood、posterior、parity/action、trajectory、显式 loss policy、独立 alias 穷举与 4 万样本 Bayes-risk calibration 均通过；全 tests/ `162 passed,4 failed`（R-N012）。 | `experiment_plan.md:307` |
| T1.3.3 | Done | 定义 regret / oracle-gap 指标。 | 完成 `physics/oracle_gap.py`、10 项 direct tests 和 `docs/tasks/T1.3.3_oracle_gap_metrics.md`；raw gaps、remaining/closed、zero/inverted/out-of-bracket、paired CI/McNemar、八类 outcome bootstrap 与 denominator reliability gate 均验证；全 tests/ `172 passed,4 failed`（R-N012）。 | `experiment_plan.md:320` |
| T1.3.4 | Done | 与已有 adaptive drifting-noise estimation baseline 对齐。 | 完成同 trace、一窗因果延迟的 Window/EKF harness；72k paired samples 上 static/Window/EKF/oracle 为 `0.06139/0.02264/0.02532/0.01139`，预注册 EKF gap closure `0.7214`、bootstrap CI `[0.7044,0.7378]`；focused `18 passed`、全 tests/ `190 passed,4 failed`（R-N012）。 | `experiment_plan.md:337` |

### Milestone 1.4：实验协议、低成本开发板与 claim contract

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T1.4.1 | Done | 冻结第一篇论文的 claim ladder。 | 完成 `docs/claim_ladder.md/.json`；五层、8 条 claim、2 条正交 lane 均有 allowed/forbidden 和升级门；当前仅 `CL1` supported，T48/T72 verdict 与 T24 原始 rows 由 8 项 tests 锁定；全 tests/ `198 passed,4 failed`（R-N012）。 | `experiment_plan.md §14.1, §14.4` |
| T1.4.2 | Done | 冻结约 300 元 FPGA 的 I/O、时钟、存储和测量边界。 | 完成 `docs/low_cost_fpga_boundary.md/.json`；冻结 Tang Nano 20K/GW2AR-18 reference target、资源/时钟/存储、5 类接口和数字测量边界；实物/报价/吞吐保持未验证，证明 115200 UART 不满足 20 ms histogram，当前 ZCU111 UIO 后端不兼容；focused 9 passed，全量 207 passed/4 failed（R-N012）。 | `experiment_plan.md §14.1` |
| T1.4.3 | Done | 冻结“两个计算域、三个时间尺度”接口。 | 完成 `docs/two_domains_three_timescales.md/.json`；2 domains、3 timescales、4 cross-domain interfaces、9 atomic steps、14 failure branches 全部可检验；direct tests 覆盖 commit/failure/queue/deadline gap；focused 11、相邻 28、全量 218 passed/4 failed（R-N012）。 | `experiment_plan.md §14.1` |
| T1.4.4 | Done | 建立 paper-parameter registry。 | 完成 `docs/paper_parameter_registry.md/.json`；8 个 source records、51 个参数（30 primary facts、3 secondary、9 assumptions、9 pending gates）逐条保存来源锚、normalization 和 transfer boundary；focused 14、相邻 42、全量 232 passed/4 failed（仅 R-N012）。 | `experiment_plan.md §14.1, §14.3, §16.2` |
| T1.4.5 | Done | 冻结 decoder/controller 与 oracle/teacher/student/bound 术语。 | 完成 `docs/decoder_controller_terminology.md/.json`；11 roles、12 artifact bindings、6 legacy aliases、10 conflation rules 锁定 input/output/objective/causality/deployability；focused 14、T1.4 相邻 56、全量 246 passed/4 failed（仅 R-N012）。 | `experiment_plan.md §15.1, §16.1` |

---

## Phase 2：实验式、多保真数字孪生

### Milestone 2.0：sBs-first 实验协议层

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T2.0.1 | Done | 冻结主/次参考协议。 | 完成 `docs/protocol_hierarchy.md/.json`：sBs 为唯一主数字孪生，sharpen–trim 为一手交叉验证，Knill/qunaught 与 ME/P-Steane 限 secondary；12 项 direct tests 锁定 cycle/观测/action/unmodeled 与禁止混用。 | `experiment_plan.md §14.1, §16.2` |
| T2.0.2 | Done | 实现 sBs Kraus/error-space transition model。 | 完成 `physics/sbs_error_space.py` grouped CPTP instrument：支持四 branch、多 `C_i` 同层 topology、逐级 trickle-down 与 Pauli frame；21 项 direct tests 覆盖 completeness/Choi/no-error/random density/Monte Carlo/失败分支。 | `experiment_plan.md §14.3` |
| T2.0.3 | Done | 实现 g/e/leakage 观测与 reset model。 | 完成 `physics/sbs_observation_reset.py`：ideal/hidden/observed/reset 四层、full 4×3 confusion、f/higher reset、X/Z e-run、leakage streak 与 deployable/truth schema；20 项 direct tests。 | `experiment_plan.md §14.1, §14.3` |
| T2.0.4 | Done | 实现实验式 cycle state machine。 | 完成 18-phase Table S3 constituent FSM、X→Z full cycle、observed/reset/VR 接线和 Pauli frame；focused 13 passed，所有 trace 标为文献参考、非目标板实测。 | `experiment_plan.md §14.3` |
| T2.0.5 | Done | 复现位移故障注入的 syndrome 趋势。 | 完成 T2.0.2/T2.0.3 接线的 9 幅度×4096-shot sweep；`l_S/4` observed e-run `4.883 [4.846,4.919]`，双边 Spearman `1/-1`，10 个预注册门全 PASS，26 项 direct tests。 | `experiment_plan.md §14.3` |
| T2.0.6 | Done | 复现 syndrome occupancy 与 correlation 趋势。 | 600×1200-cycle hidden `0.813565` vs observed-only `0.813524`；leakage removal 前后 tail `0.002976/-0.000192`，paired CI `[0.001684,0.005058]`；29 项 direct tests。 | `experiment_plan.md §14.3` |

### Milestone 2.1：混合状态 syndrome-level 快速仿真器

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T2.1.1 | Done | 实现连续漂移 + 离散 regime 的 syndrome stream generator。 | 完成完整 `DriftState` causal stream、analog/residual、X/Z `g/e/leakage`/phase、hidden regime、recovery depth 和 logical truth；20k mixture、8k loss、5k confusion 长跑及 deadlock/单轴/schema/seed 负测共 21 项通过。 | `experiment_plan.md:362`, `experiment_plan.md §14.1` |
| T2.1.2 | Done | 实现多轮控制 memory。 | 完成 observed-only nearest-lift memory，跟踪 residual/correction/confidence/Pauli+phase frame/e+leakage runs/bank version/deadline；修正 fast-loop correction 符号，26 项 direct tests 与真实 ParamBank commit 通过。 | `experiment_plan.md:377`, `experiment_plan.md §14.1` |
| T2.1.3 | Done | 实现高速 Monte Carlo 与 rare-event 模式。 | 完成 trajectory 向量化 stateful core、target-weighted burst/leakage strata、cluster CI 与 zero-event bound；production 真跑 1,000,000 cycles，约 `3.95e6 host cycles/s`，focused 19、全量 439 passed/4 failed（R-N012）。 | `experiment_plan.md:387`, `experiment_plan.md §14.3` |

### Milestone 2.2：finite-energy effective simulator

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T2.2.1 | Done | 加入有限 squeezing 等效噪声。 | 完成 channel/data-GKP/ancilla-measurement/envelope 五项分解、非高斯 lattice envelope、analytic budget 和 6×250k sweep；excess trace 严格降至 0，broad/ideal `P_L=0.031260/0.000424`，focused 25、全量 466 passed/4 failed（R-N012）。 | `experiment_plan.md:397` |
| T2.2.2 | Done | 加入 sBs/sharpen–trim 的辅助态、读出与 reset 误差。 | 完成 sBs stage-resolved bit/phase/readout overlay 与 sharpen--trim 四轮 `+y/-y` native state machine；27 项 direct、80k+80k production、schema/nonmixing/secondary non-execution 全通过；全量 494 passed/4 failed（R-N012）。 | `experiment_plan.md:416`, `experiment_plan.md §14.1, §16.2` |
| T2.2.3 | Done | 加入控制与 active-correction imperfection。 | 完成 AWG/DAC codes、affine/stochastic pulse、latency drift/diffusion、virtual rotation、两种 action order 与 exact moments；33 direct、100k production 全通过；全量 528 passed/4 failed（R-N012）。 | `experiment_plan.md:420` |

### Milestone 2.3：Fock-space 独立验证器与中保真度代理

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T2.3.1 | Done | 构造 finite-cutoff approximate GKP density-matrix model。 | 完成 normalized Hermite projection、全量 decoder→canonical dilation、displacement/loss/thermal/phase/Kerr/POVM/high-Fock；`18/24/30/36` cutoff 的最高 capture `0.99999961`、最后 embedding fidelity `0.99999882`，10 gates 全通过。 | `experiment_plan.md:436` |
| T2.3.2 | Done | 在 Fock space 中实现一轮协议对齐纠错。 | 完成 canonical analytic SBS X/Z Kraus、raw cutoff defect + shared CPTP completion、hidden/observed/action/frame/projection、exact 16 branches、100k MC 与五点 cutoff；坐标回灌后 clean conditional/survival `0.999953/0.969508`，16 gates 全通过。 | `experiment_plan.md:447`, `experiment_plan.md §14.1` |
| T2.3.3 | Done | 执行跨保真度交叉验证。 | 完成 3/5/8/10/12 dB 四 lane；T-RISK 回灌后 canonical Fock q/p high-dB gap `1.51e-7`，legacy ambiguous gap `>0.418` 保留；noise↔syndrome gap `3.93e-5`，五 cutoff、low-dB clipping 与四项归因、15 gates 全通过。 | `experiment_plan.md:459`, `experiment_plan.md §14.3, §16.1` |
| T2.3.4 | Done | 构造短时域可微 sBs trajectory simulator。 | 完成 joint cavity--ancilla `2N` density、15 参数显式门、Table S1 七段 CPTP idle、随机/回放 g/e、causal history-policy 与资源画像；37 direct、CPU/CUDA 各 17 gates 全通过。 | `experiment_plan.md §15.2` |
| T2.3.5 | Done | 验证 Feedback-GRAPE 随机轨迹梯度。 | 完成 exact reward/score 分解、常数 baseline、total/reward/score 分项 FD、四步长 sweep 与 12,288 trajectory estimator；三类 FD relative error `<3.23e-10`，最大 MC `1.120 SE`，32 direct/15 gates 全通过。 | `experiment_plan.md §15.2` |
| T2.3.6 | Done | 完成 cutoff/batch/horizon 可行性扫描。 | 65 个真实 trajectory/backward/Adam 点覆盖 cutoff `8--48`、batch `1--576` 与 2--10 cycles；cutoff 16/batch 16 全 horizon 通过，batch 576 触发显存门，cutoff 48/batch 16 触发运行时门；40 direct 全通过。 | `experiment_plan.md §15.2` |
| T2.3.7 | Done | 复现 PRL NMF 的方向性 ranking。 | V3 strict-split 同仿真器训练 5 个 MF + 5 个 NMF agents；cutoff 12 standard/MF/NMF lifetime `2.7477/6.5347/6.7408`，NMF--MF 95% CI `[0.0842,0.3281]` 且 5/5 配对为正；cutoff 16 总排序保持，但 hidden-reset 高于 NMF，故只通过 model-specific directional ranking，不升级为机制/幅度复现。 | `experiment_plan.md §15.2` |
| T2.3.8 | Done | 实现 Heisenberg noise-transfer 中保真度代理。 | 已将 signal/fluctuation/logical jump 分离；T-RISK 回灌后 stored axes 明确为 decoder-standardized classical axes、vacuum variance=1，Fock 对照转 canonical；10/12 dB 对齐和 3 dB clipping 证否均通过。 | `experiment_plan.md §16.1` |

### Milestone 2.4：经典控制链与定点时序仿真器

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T2.4.1 | Done | 建立文献系统与本项目的双 latency budget。 | 完成两个不可组合 lane、Sivak/Puviani source-scope 算术、live YAML binding、2048/4096 B UART 下界和七类 target-board/frontend `null`；23 machine gates、22 direct、50 adjacent 全通过，禁止把外部 Virtex-6/模型/软件 assumption 写成本项目实测。 | `experiment_plan.md:484`, `experiment_plan.md §14.1` |
| T2.4.2 | Done | 加入 backlog、jitter 和 deadline modeling。 | 完成真实 scheduler/ParamBank 路径上的 7 场景 × 8 paired seeds × 64k cycles；逐 seed 检测 deadline/burst/pause/conflict/FIFO，第二 writer fail closed，13 gates、67 direct+adjacent tests 通过，并量化 LER 与 fast/fresh/end-to-end availability；保持 software stress 非板测边界。 | `experiment_plan.md:505` |
| T2.4.3 | Done | 加入 fixed-point/LUT/parameter-bank error model。 | 完成 6-axis OAT × 6 levels、5 joint profiles、4 bank faults、8 paired seeds 的 368-run 位级扫描；11 gates、63 direct+adjacent、figure 5 gates 通过，生成 JSON/CSV/SVG/PDF/TIFF/PNG；资源严格限定为 bit proxy，LUT/BRAM/DSP/Fmax 未综合。 | `experiment_plan.md:509` |

---

## Phase 3：协议原生强 baseline 体系

### Milestone 3.1：静态与 oracle baseline

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T3.1.1 | Done | 实现 standard binning。 | observed-only fixed half-cell decision、hidden-truth evaluator、72k 五方法同 trace 主比较、future T5 schema guard 和 legacy P4 non-alias 完成；10 gates、92 focused+adjacent 通过。 | `experiment_plan.md:532` |
| T3.1.2 | Done | 实现 static MAP。 | total-covariance training-state average、frozen observed-only decoder、active/future schema 接入与 8-seed 576k paired validation 完成；旧 EKF strong gate 已证据降级。 | `experiment_plan.md:536` |
| T3.1.3 | Done | 接入 oracle MAP。 | exact-state/regime oracle 接入 active/future schemas；4-regime 320k 上界和 8k-cycle hidden leakage flag/cost envelope 完成，始终 nondeployable。 | `experiment_plan.md:544`, `experiment_plan.md §14.1` |
| T3.1.4 | Done | 实现 static finite-energy/protocol-aware optimized decoder。 | 完成 exact stationary hidden-carry marginalization、9×4 sBs observation/reset posterior LUT、observed-leakage-only fallback 与 4×8×20k Markov 验证；3 个非 control 场景 resolved，branch target 与 logical decoder schema 分离。 | `experiment_plan.md:548` |
| T3.1.5 | Done | 实现 top-K lattice-coset truncated MAP baseline。 | 完成四陪集 joint alias prefix sum、K=1--128、6 场景 288k full-MAP 对照和未综合 deterministic cost；收敛 K=2--4，K=128 全饱和，明确非 surface K-MWM。 | `experiment_plan.md §16.1` |

### Milestone 3.2：多轮、自适应与事件 baseline

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T3.2.1 | Done | 实现 memory-assisted Bayesian decoder。 | 完成 20-cycle observed-only joint periodic posterior、same-prior final-outcome static comparator、4,096-episode Student-t/proper-score 验证、128/256 grid convergence 与 task-specific registry role 修复；明确非 Wan finite-energy fidelity/FPGA reproduction。 | `experiment_plan.md:558` |
| T3.2.2 | Done | 实现 EWMA / Kalman adaptive MAP。 | 完成 full-covariance circular-moment window/EWMA/constant-velocity Kalman、4 类连续漂移 8-seed 157 万 paired samples、training-only 网格选择与 15 项门禁；两种 adaptive 均逐场景显著优于 formal static，latest-window 强 comparator 保留，硬件字段为 null。 | `experiment_plan.md:562` |
| T3.2.3 | Done | 实现 sliding-window syndrome estimator。 | 完成 384--1536 六窗、96-sample circular-feature chunk 增量统计、training-only selection、4 场景 8-seed 157 万 paired samples 与成本面；训练/evaluation aggregate 均选择 384，长 uniform history 的全局收益被证否，局部场景最佳仅作 diagnostic。 | `experiment_plan.md:577` |
| T3.2.4 | Done | 实现 post-selection 诊断上界与成本核算。 | 完成 training-only posterior-risk thresholds、99.5%--50% survival、observed/random/truth-upper 三 lane、4 档 rejection penalty、256-row Source Data；90% conditional LER 大降但 unit-rejection total cost 显著高于 raw，明确不进入在线主增益。 | `experiment_plan.md:581`, `experiment_plan.md §14.4` |
| T3.2.5 | Done | 实现 run-length FSM / parameter-bank baseline。 | 完成 3-bit 饱和 counters、五态优先级、phase tie-break、真实双 bank 原子同步、24-grid/384k same-trace event-cost 验证与持续 local-safe conflict path；非退化 FSM 优于 static 但显著弱于 memoryless，负结果保留且不冒充 LER/RTL。 | `experiment_plan.md §14.1, §14.3` |
| T3.2.6 | Done | 实现 HMM 或 change-point regime baseline。 | 完成 observed-only 四状态 causal Gaussian HMM、same-emission memoryless ablation、3/3/8 disjoint seeds、54×10 training-only 网格、4,096-window evaluation、proper-score/calibration/delay/false-switch 指标与 896-float/800-MAC shared future-CNN budget；明确仅为 synthetic host estimator。 | `experiment_plan.md §14.3` |
| T3.2.7 | Done | 实现 latest-outcome FNN / Markovian feedback。 | 完成 72,853-param/72,266-MAC exact latest-token FNN、5-agent 300-epoch strict split、同 trace frozen NMF/旧 MF、18,023-row Source Data；cutoff 12 无显著 memory gain、cutoff 16 排序反转，禁止稳定机制 claim。 | `experiment_plan.md §15.3` |
| T3.2.8 | Done | 实现 autonomous sBs 物理时间 baseline。 | 已完成 7/10 us protocol-native nonselective 两路径、共同 700 us、cutoff12/16×三噪声、1,020 cycles、raw event ledger 与 4,362-row Source Data；6/6 per-cycle autonomous 更好而 6/6 per-us 更差，禁止 lifetime 缩放、单 endpoint 或目标板 timing claim。 | `experiment_plan.md §15.3` |
| T3.2.9 | Done | 实现有限时域 trajectory lookup control oracle。 | 已完成两-cycle 15-node/225-scalar causal tree、exact 16-branch objective、optimized open-loop 嵌套、两 families×3 restarts×`300+250` epochs、cutoff16 frozen transfer、3,418-row Source Data 与指数资源审计；仅作为 empirical nondeployable control reference。 | `experiment_plan.md §15.1, §15.3` |
| T3.2.10 | Done | 实现 PRL-inspired 指数递推 baseline。 | 完成 75-trainable/105-stored causal recurrence、3×`300+250` exact optimization、cutoff12/16 frozen transfer、Q mirror、72-grid/384k same-trace event comparison 与 1,888-row Source Data；cutoff12 位于 standard/lookup 之间，event 优于 FSM 但弱于 memoryless，物理 leakage/RTL claim 保持关闭。 | `experiment_plan.md §15.3, §15.4` |
| T3.2.11 | Done | 执行 memory-specific 消融。 | 完成 5 个 frozen NMF×prefix-consistent shuffle/truncation/reset/latest-only、5 个同预算重训 latest-only、cutoff12/16 闭环重放、28,230-row Source Data；两个 cutoff 均未通过四对照联合机制门，保留 signed reversal。 | `experiment_plan.md §15.3` |

---

## Phase 4：两个计算域、三个时间尺度控制器

### Milestone 4.1：主机慢回路状态估计与离线 teacher

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T4.1.1 | Done | 在匹配预算下选择慢回路模型。 | 完成共同 8-window four-regime task 下 TCN/GRU/HMM/Kalman/指数递推/FSM 六族 validation-only 选型；HMM validation/evaluation NLL `0.454975/0.455711`，rolling cache 926 MAC/3728 B 常驻/104 B scratch；13 gates、33 focused，24,240-row Source Data。 | `experiment_plan.md:593`, `experiment_plan.md §15.1` |
| T4.1.2 | Done | 定义实验式 history 输入。 | 完成 256-cycle×53-feature observed-only schema、真实 syndrome/FSM/LLR/scheduler producer adapter、padding/saturation/provenance 与递归 truth-leak audit；17 gates、16,384-row Source Data。 | `experiment_plan.md:604`, `experiment_plan.md §14.1` |
| T4.1.3 | Done | 定义混合状态输出。 | 完成连续 9 参数、四态 posterior、leakage/recovery burden、9×9 block-bootstrap uncertainty、future K/b recommendation；17 gates、456-row Source Data、58 次精确 atomic commit。 | `experiment_plan.md:621`, `experiment_plan.md §14.1` |
| T4.1.4 | Done | 定义多目标 loss 与 calibration。 | 完成六项 loss、3/2/3 strict split、temperature/uniform-mix/uncertainty/fallback calibration、19 gates 与 448-row Source Data；proper scores 改善但 false fallback `1.0` 负结果保留。 | `experiment_plan.md:641` |
| T4.1.5 | Done | 实现 offline teacher / online student 分离。 | 完成 5-agent frozen teacher hash 恢复、3-split/3-restart 75 参数递推蒸馏、105-scalar online artifact、21 gates 与 15,360-row Source Data；evaluation imitation MSE `1.453624e-6`，严格保留 physical-gain 未验证边界。 | `experiment_plan.md §14.1, §14.4, §15.1` |

### Milestone 4.2：FPGA 逐周期确定性 fast path

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T4.2.1 | Done | 设计 parametric MAP-LUT。 | 完成 active K/b effective-model 反解、X/Z Q9.12 ROM、10-bit ADC/8-bit address half-bin interpolation、20 gates 和 16,384-row exhaustive Source Data；hard action mismatch 0，5-cycle/II=1 仅为 software pipeline contract。 | `experiment_plan.md:679` |
| T4.2.2 | Done | 设计实验式事件状态机和硬件动作。 | 完成六态 observed-event FSM、六个 3-bit 饱和计数器、leakage/reset handshake、Pauli/phase-frame 与 1,024-row replay；20 gates，MAP+event 固定 6 cycles/II=1，仅为 software contract。 | `experiment_plan.md:695`, `experiment_plan.md §14.1` |
| T4.2.3 | Done | 设计 conservative fallback 和健康标志。 | 完成 controller-owned trusted image registry、14-bit fault taxonomy、frame-hold/reset、4,096-row replay 与 reason trace；20 gates 覆盖 OOD/leakage/stale/CRC/SHA/version/deadline/MAP/ack。 | `experiment_plan.md:705` |
| T4.2.4 | Done | 完成 fast path 定点化。 | 完成 MAP→health→event→frame 全 word/rounding contract、四档 precision×8-bank 87,040-code audit 与 128-cluster paired LER；21 gates，selected ΔLER CI 跨零。 | `experiment_plan.md:709` |

### Milestone 4.3：三时间尺度闭环与安全更新

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T4.3.1 | Done | 定义 fast/event/slow 三种更新频率。 | 明确每子周期动作、窗口级 health update、跨 run/分钟级重标定；量化 adaptation lag。 | `experiment_plan.md:734`, `experiment_plan.md §14.1` |
| T4.3.2 | Done | 实现双参数库、原子切换和 hysteresis。 | 完成 full-image canonical payload、manifest/payload/image CRC/SHA、version/timestamp/CAS/anti-replay、两窗 hysteresis、safe-boundary atomic switch、ack/readback；7518-row evidence 穷举全部 3745 prefix 与 3745 byte flip，17/17 gates。 | `experiment_plan.md:747` |
| T4.3.3 | Done | 测试闭环稳定性和故障恢复。 | 完成 8 场景×4 seeds×23996 cycles；767872 个逐周期 action 无 undefined/blocking-correction/frame-overflow，ack uncertainty、freshness refresh、guard 与 monotonic LKG republish 可追溯；17/17 gates。 | `experiment_plan.md:760`, `experiment_plan.md §14.3` |

### Milestone 4.4：Non-Markovian teacher-to-student 蒸馏

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T4.4.1 | Done | 训练 bounded residual RNN/GRU teacher。 | 完成 3 个 fresh 72,853 参数 GRU restart×320 epochs、旧 checkpoint state/seed non-reuse、15-output nominal+hard-bound 动作、strict split/validation-only 选模、cutoff12/16 held-out 与 1,074-row Source Data；21/21 gates，primary score gain `0.253603`，两次 cap hit 显式保留。 | `experiment_plan.md §15.2, §15.4` |
| T4.4.2 | Done | 分析 teacher hidden state 和控制轨迹。 | 完成 10×128 native g/e、leakage OOD proxy、20-half-cycle forced p(g)、24/8 strict-split probe、PCA、30 参数指数拟合、双向 impulse/Jacobian 与 2,089-row Source Data；17/17 gates，hidden/control 95% PC 均1，control 1% memory 10/12。 | `experiment_plan.md §15.3, §15.4` |
| T4.4.3 | Done | 拟合指数递推或有限状态 student。 | 完成 1/2/4-state×3-restart strict-split 蒸馏、validation-only 选维、58,356-row Source Data 和 pure-NumPy fail-closed artifact；选中 4 states/95 scalars，held-out MSE `6.083136e-6`，16/16 gates。 | `experiment_plan.md §15.4` |
| T4.4.4 | Done | 执行 teacher-student gain-retention gate。 | 完成 10-cycle 新 paired seeds×双 cutoff、全部5个 MF、teacher/handcrafted/student、独立2-cycle exact oracle、显式 burden/cost 与448-row Source Data；18/18 gates，最低 point/CI-lower retention `0.981457/0.944501`。 | `experiment_plan.md §15.3, §15.4` |
| T4.4.5 | Done | 冻结 strong/falsified 双分支。 | 完成 72 parent gates、4 implementation hashes、7 file bindings、8 predicates 与112-row ledger 的 fail-closed 判定；激活 qualified student-retention，保留 MF reversal/七类禁止 claim，任一 parent/T5/T6 gate 失败自动回退 MAP-LUT。 | `experiment_plan.md §15.5` |

---

## Phase 5：分层证据门与核心结论验证

任何主张只有通过对应 evidence gate 后才能进入论文正文；未通过时执行失败分支，不用语言包装替代证据。

### Milestone 5.0：协议与数字孪生可信度门

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T5.0.1 | Done | 建立文献趋势 reproduction table。 | 完成 7-source、8-anchor、6-artifact、14-target registry；逐行固定主线/secondary、数值/方向容差、calibration/holdout、status 与禁止迁移；17/17 gates、52-row Source Data、224 adjacent tests 通过，5 个 pending 未冒充复现。 | `experiment_plan.md §14.3, §15.2, §16.2` |
| T5.0.2 | Done | 执行跨保真度和独立 holdout 验证。 | calibration/pilot/formal points 完全隔离；main cross-fidelity 因 `10.25 dB` pooled z=`2.293338>2` 判 FAIL 并保留，secondary P-Steane 252-point analytic holdout PASS；291-row Source Data、118 focused、319 adjacent tests 通过。 | `experiment_plan.md §14.3` |

### Milestone 5.1：算法与 oracle-gap 门

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T5.1.1 | In Progress | 建立完整 comparison set。 | 比较 no correction、standard/autonomous sBs、static/top-K/decoder-oracle MAP、finite-energy static、Bayesian、EWMA/Kalman、window、run-length、HMM、MF/FNN、指数递推、RNN teacher、distilled student 和 control oracle；Knill/P-Steane 不进入 sBs 主排名。 | `experiment_plan.md:779`, `experiment_plan.md §14.3, §15.3, §16.2` |
| T5.1.2 | Todo | 运行混合 noise/regime scenario matrix。 | 覆盖 static Gaussian、mean/variance/correlation drift、loss、readout/ancilla drift、burst/outlier、large-error recovery、leakage 和 calibration shift。 | `experiment_plan.md:792`, `experiment_plan.md §14.1` |
| T5.1.3 | Todo | 报告 average、tail 和双 oracle-gap 指标。 | 报告 `P_L`、worst/95% window LER、decoder-oracle gap、短时 control-oracle gap、bootstrap CI、paired seeds 和多重比较策略。 | `experiment_plan.md:805`, `experiment_plan.md §15.1` |
| T5.1.4 | Todo | 执行成功/证否分支。 | static 下不退化；drift/regime 下至少对强可部署 baseline 有可重复优势；否则改为事件感知 adaptive MAP/FPGA co-design 论文，不保留 CNN 性能主张。 | `experiment_plan.md:832` |
| T5.1.5 | Todo | 执行物理时间与控制成本公平化。 | 同时报 per-cycle、per-microsecond、measurement/reset 次数、active-control 次数和 classical latency；autonomous/feedback 周期差异不能隐去。 | `experiment_plan.md §15.3` |
| T5.1.6 | Todo | 报告实验可行性约束。 | 报告 `p(g)`、e/leakage occupancy、reset burden、parameter slew/saturation、fallback 和 unsafe-action rate；峰值 lifetime 不得覆盖不可部署代价。 | `experiment_plan.md §15.3` |

### Milestone 5.2：因果故障注入与 syndrome 诊断门

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T5.2.1 | Todo | 注入 displacement / large-distance error。 | recovery depth、e-run、逻辑失败率随到最近逻辑操作距离呈可解释趋势。 | `experiment_plan.md §14.3` |
| T5.2.2 | Todo | 分别注入 ancilla bit/phase flip 与 readout error。 | 敏感度方向与协议设计一致；不得用同一扰动同时改变多个通道后声称因果。 | `experiment_plan.md §14.3` |
| T5.2.3 | Todo | 注入 leakage 与 reset failure。 | 报告 detection delay、false alarm、correlation tail、availability 和 recovery cost；做 leakage-free 消融。 | `experiment_plan.md §14.3` |

### Milestone 5.3：逻辑通道、coherence gain 与成本门

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T5.3.1 | Todo | 重构 logical channel。 | 对六个 Pauli eigenstates 或等价 PTM 数据报告 Pauli lifetimes、PTM、non-Pauli/leakage 诊断；QEC on/off 同时报 per-cycle 与 wall-clock lifetime，不以跨平台 cycle time 替代本项目时序。 | `experiment_plan.md:856`, `experiment_plan.md §14.3, §16.2` |
| T5.3.2 | Todo | 报告 `F_avg`、`F_e` 与短时有效退极化率。 | 不用不适用的单指数拟合替代平均通道定义；不确定度可追溯。 | `experiment_plan.md:869`, `experiment_plan.md §14.3` |
| T5.3.3 | Todo | 定义 simulated break-even / operational boundary。 | 仅在同一模型和成本口径下比较 passive/uncorrected 与 active logical channel；写作中使用 simulation-derived coherence gain。 | `experiment_plan.md:983-1018`, `experiment_plan.md §14.4` |
| T5.3.4 | Todo | 核算真实纠错和 post-selection 成本。 | 计入 repeats、active pulses、rejection/survival、squeezing、classical resource、latency 和 achieved LER/fidelity。 | `experiment_plan.md:1018`, `experiment_plan.md §14.4` |
| T5.3.5 | Todo | 计算 QEC-matrix/Petz near-optimal channel-recovery bound。 | 小 cutoff 与 SDP optimal recovery 校验双边界；可行时扩展到更高能量/更大 cutoff；报告实际 sBs、teacher/student 到该 bound 的 gap，并明确它是假设任意恢复的编码—噪声性能界而非可部署 decoder。 | `experiment_plan.md §16.1` |

### Milestone 5.4：鲁棒性、消融与负结果门

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T5.4.1 | Todo | 运行 held-out/OOD 测试。 | 覆盖未见 drift family、参数范围、leakage rate、measurement confusion 和通信扰动。 | `experiment_plan.md:900` |
| T5.4.2 | Todo | 验证 uncertainty-gated fallback。 | 相对不带 fallback 的系统降低 catastrophic failure，同时报告不必要 fallback 率和性能代价。 | `experiment_plan.md:910` |
| T5.4.3 | Todo | 完成因果消融和负结果表。 | 分别关闭 history、CNN residual、regime state、run-length、parameter update、fallback；保留失败场景和 claim 降级决定。 | `experiment_plan.md:914`, `experiment_plan.md §14.3` |
| T5.4.4 | Todo | 审计 multi-agent/seed 选择偏差。 | 报告全部 agents/seeds、median/IQR/worst quartile；模型只按 validation 选择，独立 test 不参与 best-agent post-selection。 | `experiment_plan.md §15.3` |
| T5.4.5 | Todo | 验证训练 horizon 到长时部署的外推。 | 扫描训练 horizon；检查 hidden-state boundedness、reset sensitivity 和在 `1e3/1e5/1e6` cycles 的性能与数值稳定性。 | `experiment_plan.md §15.3` |
| T5.4.6 | Todo | 运行 randomized model-mismatch family。 | 覆盖随机 gate bias、readout confusion、leakage/reset failure、cavity dephasing、drift 和 unseen timing/dynamics；不得只报告单一 bias vector。 | `experiment_plan.md §15.3` |

### Milestone 5.5：硬件设计冻结门

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T5.5.1 | Todo | 冻结 Python bit-accurate hardware reference。 | 输入/输出、位宽、饱和、舍入、FSM、parameter-bank 和 trace schema 固定，作为 RTL golden model。 | `experiment_plan.md:942`, `experiment_plan.md §14.3` |
| T5.5.2 | Todo | 运行目标器件 synthesis / timing estimate。 | 报告 Fmax、LUT/FF/BRAM/DSP、critical path 和估计 latency；明确尚非板级实测。 | `experiment_plan.md:946` |
| T5.5.3 | Todo | 做 precision-resource-performance Pareto 选择。 | 联合扫描位宽、top-K、student 状态维数和并行度，报告 LER/gain、LUT/FF/BRAM/DSP、Fmax、latency；选择能装入实际廉价板卡且满足 deadline 的单一部署点，装不下则缩小模型。 | `experiment_plan.md:960`, `experiment_plan.md §14.1, §16.1` |
| T5.5.4 | Todo | 比较完整 GRU、量化 GRU 与蒸馏 student 的硬件可行性。 | 报告参数/权重存储、MAC、BRAM/DSP、Fmax、worst-case latency 和 gain retention；默认 student 主线，完整 GRU 只有综合通过才进入增强路线。 | `experiment_plan.md §15.4` |

---

## Phase 6：低成本 FPGA 实测与 hardware-in-the-loop

### Milestone 6.1：板卡与通信链路 bring-up

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T6.1.1 | Todo | 锁定并记录实际开发板。 | 记录型号、采购成本、FPGA 器件、工具链版本、时钟、供电和接口；照片与版本信息可用于论文 supplement。 | `experiment_plan.md §14.1` |
| T6.1.2 | Todo | 实现可复现 syndrome replay 协议。 | PC 通过可用 UART/USB/JTAG 发送带 sequence/version/CRC 的定点 I/Q 或分类后 syndrome；支持错误注入和流控。 | `experiment_plan.md §14.1, §14.3` |
| T6.1.3 | Todo | 建立板级时间戳和测量方法。 | 明确 on-chip cycle counter、logic analyzer/GPIO 测量、host timestamp 的用途和分辨率；区分 transport latency 与 core latency。 | `experiment_plan.md §14.3` |

### Milestone 6.2：FPGA fast path 原型

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T6.2.1 | Todo | 实现定点 MAP-LUT 与事件 FSM。 | RTL/HDL 实现 syndrome classification、MAP-LUT、run-length、frame accumulator、fallback 和 action 输出。 | `experiment_plan.md §14.1` |
| T6.2.2 | Todo | 完成 testbench 与 Python golden 对齐。 | 正常、边界、饱和、leakage、CRC/version、reset、deadline cases bit-for-bit 一致；记录覆盖率。 | `experiment_plan.md §14.3` |
| T6.2.3 | Todo | 完成板级 correctness smoke。 | 实际板上回放固定 trace，输出与 RTL simulation/Python golden 一致；失败可通过 trace 定位。 | `experiment_plan.md §14.3` |
| T6.2.4 | Todo | 实现定点蒸馏递推 student。 | RTL/HDL 支持 g/e/leakage 分支的低维递推、参数安全包络、饱和和 fallback；与 Python student bit-for-bit 一致。 | `experiment_plan.md §15.4` |
| T6.2.5 | Todo | 评估 optional quantized GRU datapath。 | 仅在 T5.5.4 证明目标器件可装入并满足 deadline 后实现；否则记录为 Dropped，不影响 student 主线。 | `experiment_plan.md §15.4` |

### Milestone 6.3：主机慢回路与 HIL 闭环

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T6.3.1 | Todo | 实现 host estimator -> FPGA parameter update。 | 默认路线为 PC/CPU/GPU 慢回路更新双参数库；quantized CNN 上板仅作可选增强，不进入 critical path。 | `experiment_plan.md §14.1, §14.4` |
| T6.3.2 | Todo | 运行静态、漂移与 regime HIL matrix。 | 对 MAP-LUT、MF、teacher replay、distilled student 记录 input、history state、bank version、LLR/control action、fallback reason 和 latency。 | `experiment_plan.md §14.3, §15.4` |
| T6.3.3 | Todo | 与 Python fixed-point/oracle 离线结果对齐。 | HIL 与 Python fixed-point 在相同 trace 上 bit-accurate；性能差异可归因于 transport/deadline/fallback，而不是隐藏处理。 | `experiment_plan.md §14.3` |

### Milestone 6.4：长时、资源与失败实验

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T6.4.1 | Todo | 运行至少 `1e5`、目标 `1e6` cycles 长序列。 | 覆盖稀有 leakage、burst、计数器/hidden-state 饱和、parameter update、通信停顿；报告状态有界性和零/非零 deadline miss 的置信上界。 | `experiment_plan.md §14.3, §15.3` |
| T6.4.2 | Todo | 测量板级 latency、jitter、throughput 和资源。 | 报告 core/transport/end-to-end 三种延迟、worst case、Fmax、LUT/FF/BRAM/DSP；可测时补充功耗。 | `experiment_plan.md §14.3` |
| T6.4.3 | Todo | 运行故障恢复和 negative-path 实验。 | CRC 错误、stale bank、host timeout、FIFO overflow、reset storm 下无未定义动作；形成失败模式表。 | `experiment_plan.md §14.3` |

---

## Phase 7：论文、审稿风险与可复现发布

### Milestone 7.1：主张—证据—主图冻结

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T7.1.1 | Todo | 冻结 claim-evidence-boundary matrix。 | 每个 abstract/conclusion claim 映射到 figure/table/data/code 和证据层级；无证据 claim 删除或降级。 | `experiment_plan.md §14.3, §14.4` |
| T7.1.2 | Todo | 冻结主图 1—2。 | Fig.1：实验边界、POMDP/belief-state 与三时间尺度架构；Fig.2：sBs 数字孪生、Feedback-GRAPE teacher 和 distilled FPGA student。 | `experiment_plan.md §14.3, §15.5` |
| T7.1.3 | Todo | 冻结主图 3—4。 | Fig.3：standard/autonomous/MF/NMF/student/control-oracle 的决定性性能、OOD 与故障注入；Fig.4：策略解释、gain retention 和板级 latency/resource/HIL。 | `experiment_plan.md §14.3, §15.5` |
| T7.1.4 | Todo | 冻结 Supplement figure contract。 | 将 cutoff/gradient、noise-transfer 有效域、Petz bound、top-K Pareto、secondary protocol reproduction、六 Pauli states、all-seed distributions、完整 OOD、fixed-point 和 failure modes 放入 Supplement，主文保持单一论证主线。 | `experiment_plan.md §15.5, §16.2` |

### Milestone 7.2：论文正文与补充材料

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T7.2.1 | Todo | 起草 Introduction 与 Related Work。 | 叙事聚焦 experiment-informed classical control / decoding gap，不声称替代真实量子实验。 | `experiment_plan.md:1070`, `experiment_plan.md §14.4` |
| T7.2.2 | Todo | 起草 Methods。 | 覆盖协议数字孪生、混合噪声、baseline、三时间尺度控制、定点/RTL/HIL 和统计方法，可复现。 | `experiment_plan.md:1070`, `experiment_plan.md §14.3` |
| T7.2.3 | Todo | 起草 Results。 | 依证据门顺序报告 protocol、causal injection、algorithm、logical channel、robustness、board/HIL；负结果不隐藏。 | `experiment_plan.md:1070`, `experiment_plan.md §14.3` |
| T7.2.4 | Todo | 起草 Discussion/Conclusion。 | 明确无 cavity/transmon、无真实 beyond-break-even、无板上训练；讨论成本、外部效度和真实实验接入路径。 | `experiment_plan.md:1070`, `experiment_plan.md §14.4` |
| T7.2.5 | Todo | 完成 Supplementary。 | 包含公式、参数表、完整 baseline、置信区间、消融、失败模式、RTL/工具链、长序列和复现说明。 | `experiment_plan.md:1070` |

### Milestone 7.3：审稿风险预处理

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T7.3.1 | Todo | 回答“为何不用 exact/oracle MAP？” | oracle 是不可部署上界；贡献限定为 drift/regime 下缩小 static-to-oracle gap。 | `experiment_plan.md:1099` |
| T7.3.2 | Todo | 回答“CNN 是否只是过拟合模拟器？” | 用强 baseline、held-out protocol/range、跨保真度、消融和 board HIL 回答；证据不足则删除 CNN 主张。 | `experiment_plan.md:1099` |
| T7.3.3 | Todo | 回答“为何称实验相关但没有量子硬件？” | 清楚区分 literature fact、digital twin、board measurement 和 HIL；标题摘要不得使用 experimental GKP QEC。 | `experiment_plan.md §14.4` |
| T7.3.4 | Todo | 回答“post-selection/break-even 是否夸大？” | 主指标不依赖 post-selection；仅报告 simulation-derived coherence gain 和完整成本。 | `experiment_plan.md §14.4` |
| T7.3.5 | Todo | 回答“是否只是复现 NMF PRL？” | 新贡献必须落在 leakage/drift/model mismatch、teacher-to-student compression、低成本 FPGA deadline/HIL 和严格选择偏差审计；缺一则降低 novelty claim。 | `experiment_plan.md §15.5` |
| T7.3.6 | Todo | 回答“完整 RNN 是否真的能实时上板？” | 以 T5.5.4/Phase 6 实测区分 full GRU、quantized GRU 和 distilled student；不能把 teacher GPU 推理写成 FPGA critical path。 | `experiment_plan.md §15.4` |

### Milestone 7.4：可复现发布

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T7.4.1 | Todo | 冻结数据、配置和 provenance。 | 每张主图可追溯到 raw/processed data、config、seed、commit、板卡/工具链版本。 | `experiment_plan.md §14.3` |
| T7.4.2 | Todo | 提供一键复现实验和 artifact manifest。 | 软件仿真可一键运行；硬件结果提供 trace、golden output、bitstream/source 或可审计替代物。 | `experiment_plan.md §14.3` |
| T7.4.3 | Todo | 完成投稿前证据审计。 | claim-evidence matrix、任务记录、风险表、主文、补充材料和仓库状态一致，无“estimate 写成 measured”。 | `experiment_plan.md §14.4` |

---

## Phase 8：可选真实 GKP 数据或量子硬件接入

本阶段不阻塞第一篇论文，不得在尚未获得访问权限或真实数据前写成承诺性主结果。

### Milestone 8.1：真实数据离线验证

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T8.1.1 | Todo | 合法收集带 metadata 的真实 GKP syndrome history。 | 数据许可、协议、时间戳、观测语义和可用 label/tomography 信息明确。 | `experiment_plan.md §14.2` |
| T8.1.2 | Todo | 从真实 syndrome history 估计 drift/regime。 | 给出 uncertainty 和不可辨识性，不把离线估计写成在线闭环。 | `experiment_plan.md §14.2` |
| T8.1.3 | Todo | 离线比较 baseline 与 proposed decoder。 | 仅在 label/tomography 支持时报告 LER/fidelity；否则报告 calibration/diagnostic 指标。 | `experiment_plan.md §14.2` |

### Milestone 8.2：可选控制链路接入

| ID | 状态 | 任务 | 产物 / 通过标准 | 来源 |
| --- | --- | --- | --- | --- |
| T8.2.1 | Todo | 接入真实 digitized measurement/control stream。 | 先完成 shadow/pseudo-closed-loop，无权改变真实实验动作时不越权。 | `experiment_plan.md §14.2` |
| T8.2.2 | Todo | 运行 Pauli/phase-frame update。 | 只有在链路安全、时序和回滚通过审查后执行。 | `experiment_plan.md §14.2` |
| T8.2.3 | Todo | 最后评估 active displacement feedback。 | 仅在真实实验团队授权且 frame-update 路径稳定后尝试。 | `experiment_plan.md §14.2` |

---

## 与规划约束的交叉检查

| 约束 | v2.2 任务板处理方式 | 来源 |
| --- | --- | --- |
| 已完成证据不得因重构丢失。 | T0—T1.3.2 原状态和产物保留；只重排未开始任务。 | `docs/tasks/`, `experiment_plan.md §14.2` |
| 不声称 CNN 超过 oracle / 已知静态噪声下 MAP。 | T1.3.2—T1.3.3、T3.1.3、T5.1.4、T7.3.1 将 oracle 设为不可部署上界。 | `rough_plan.md:72`, `experiment_plan.md:320` |
| 主范围保持 single-mode approximate GKP。 | T0.1.1 保持 Done；sBs/sharpen–trim 是同一主对象的实验协议层，不引入外层码。 | `experiment_plan.md:57`, `experiment_plan.md §14.1` |
| 实验 syndrome 不能只建模为平稳连续高斯。 | T2.0.2—T2.1.3 加入 error hierarchy、g/e/leakage、recovery 和离散 regime。 | `experiment_plan.md §14.1` |
| 主机学习不进入逐周期 critical path。 | T4.1.5、T4.2、T4.3 和 T6.3 将 offline teacher、host estimator 与 FPGA fast path 分开。 | `experiment_plan.md §14.1, §14.4` |
| decoder、controller、两类 oracle 与 recovery bound 不得混称。 | T1.4.5 冻结术语；T1.3.2/T3.1.3 使用 decoder oracle，T3.2.9 使用 control oracle，T4.4 使用 controller teacher/student，T5.3.5 使用 channel-recovery bound。 | `experiment_plan.md §15.1, §16.1` |
| NMF teacher 必须先过可微仿真与梯度可行性门。 | T2.3.4—T2.3.7 依次验证 trajectory simulator、梯度、资源和方向性 ranking；任一关键门失败即回退 v2 主线。 | `experiment_plan.md §15.2` |
| 不能用 best-of-N 或短时训练外推包装稳定性。 | T5.4.4—T5.4.5 报告全部 agents/seeds、validation-based selection、horizon sweep 和长时 hidden-state 有界性。 | `experiment_plan.md §15.3` |
| teacher 不直接等于可部署 FPGA 控制器。 | T4.4.3—T4.4.4 先蒸馏低维 student；T5.5.4、T6.2.4—T6.2.5 以资源/deadline gate 决定 student、量化 GRU 或 Dropped 分支。 | `experiment_plan.md §15.4` |
| noise-transfer 代理不得替代高保真物理验证。 | T2.3.8 明确高 squeezing 有效区和 clipping 失效区；T2.3.3 必须与 Fock/effective/syndrome model 交叉验证。 | `experiment_plan.md §16.1` |
| K-MWM 与 surface-GKP 不得借机进入 single-mode 主范围。 | T3.1.5 只借鉴 top-K likelihood accumulation，形成 single-mode lattice-coset truncated MAP；不实现外层匹配图。 | `experiment_plan.md §16.1, §16.3` |
| Knill/P-Steane 与多模 trapped-ion 只作 secondary evidence。 | T2.0.1、T2.2.2、T5.0.1 和 T7.1.4 将其限制为协议趋势、解析回归和实验报告模板，不进入 sBs 主排名或 FPGA 物理 claim。 | `experiment_plan.md §16.2, §16.3` |
| baseline 必须协议原生且强。 | Phase 3 和 T5.1 覆盖 MAP、Bayesian、Kalman、window、run-length、HMM/change-point。 | `rough_plan.md:447`, `experiment_plan.md §14.3` |
| post-selection 不得冒充在线纠错。 | T3.2.4 和 T5.3.4 仅把它作为诊断上界并核算 survival cost。 | `experiment_plan.md §14.4` |
| 低成本 FPGA 不等于真实量子实验。 | T1.4、Phase 6 和 T7.3.3 分离 digital control-plane/HIL 与 cavity/transmon/microwave claim。 | `experiment_plan.md §14.1, §14.4` |
| 论文主图不得先于真实板卡证据冻结。 | v2 将低成本 FPGA/HIL 设为 Phase 6，论文和主图设为 Phase 7。 | `experiment_plan.md §14.2` |
| single-mode GKP 避免 surface-code threshold 语言。 | T5.3.3、T7.3.4 使用 operational boundary、simulation-derived coherence gain 和 logical lifetime。 | `rough_plan.md:270`, `experiment_plan.md §14.4` |

## 插入任务区

当 `docs/risks.md` 中的风险被判定为需要立即处理，且原 `experiment_plan.md` 没有对应 task 时，在这里插入 `T-RISK-YYYYMMDD-NN` 任务。插入任务不得修改 `docs/rough_plan.md`；只有真实结果与计划发生实质出入时，才低频更新 `docs/experiment_plan.md`。

| ID | 状态 | 来源风险 | 建议位置 | 任务 | 产物 / 通过标准 |
| --- | --- | --- | --- | --- | --- |
| T-RISK-20260706-01 | Done | R-008 | T0.2.2 前 | 确认并补齐计划参考池中的 Zotero 缺口；实际论文/报告条目全部补入 Zotero，工具/索引/教程类条目保留为 `工具/泛称`。 | 完成 `docs/tasks/T-RISK-20260706-01_zotero_supplement.md`、`docs/tasks/T-RISK-20260706-01_zotero_supplement.bib` 和 `docs/tasks/T-RISK-20260706-01_zotero_completion_round2.bib`；两批共补入 41 个 Zotero item key 并同步 `docs/literature_matrix.md`。 |
| T-RISK-20260712-01 | Done | R-014—R-018 | T1.3.4 后、Phase 2 前 | 根据三篇 GKP 实验/综述材料重构未开始任务，使协议、混合非平稳性、低成本 FPGA 和论文证据门闭环。 | 完成 `docs/tasks/T-RISK-20260712-01_experiment_aligned_task_board_restructure.md`；保留 T0—T1.3.2，新增 M1.4/M2.0，重构 Phase 2—8，并在 `experiment_plan.md` 第 14 节登记低频修订。 |
| T-RISK-20260713-01 | Done | R-019—R-023 | T1.4/M2.3 后、Phase 3 前 | 根据 NMF PRL 定向补强 memory-specific baseline、Feedback-GRAPE 可行性门、teacher-to-student 路线和 FPGA 部署证据门。 | 完成 `docs/tasks/T-RISK-20260713-01_nonmarkovian_teacher_student_board_strengthening.md`；新增 `experiment_plan.md` 第 15 节和 v2.1 task/风险闭环，当前推荐任务保持 `T1.3.3`。 |
| T-RISK-20260713-02 | Done | R-024—R-026 | T1.4/M2.3 后、Phase 3 前 | 根据六篇补充论文进行 v2.2 小幅补强，增加 noise-transfer surrogate、channel-recovery bound、top-K lattice-coset MAP，并限制 secondary protocol 的范围。 | 完成 v2.2 `docs/task_board.md`、`experiment_plan.md` 第 16 节、风险同步和 `docs/tasks/T-RISK-20260713-02_six_paper_v22_board_strengthening.md`；当前推荐任务保持 `T1.3.3`。 |
| T-RISK-20260714-01 | Done | R-N041 | T2.3.3 后、T2.3.4 前 | 冻结 canonical/decoder/displacement/symplectic quadrature normalization，解释并修复 Fourier-p audit。 | 完成四 chart、完整 damped-projector dilation、32 direct、15 machine gates、canonical q/p high-dB gap `1.51e-7` 与 legacy gap `>0.418`；R-N041 降为 Mitigated，joint coherent claim 仍 fail closed。 |

## 进度日志

| 日期 | Task ID | 状态变化 | 产物 / 验证 | 备注 |
| --- | --- | --- | --- | --- |
| 2026-07-06 | Board | Created | 从 `experiment_plan.md` 抽取初版任务板，并对照 `rough_plan.md` 检查。 | 所有 task 初始化为 `Todo`。 |
| 2026-07-06 | Board | Revised | 将任务板正文改为中文形式，保留 task ID、状态枚举和源文件行号。 | 文档写作优先中文。 |
| 2026-07-06 | Governance | Created | 初始化 `README.md`、`.gitignore`、`AGENTS.md`、`docs/tasks/README.md` 和 `docs/risks.md`。 | 明确 `rough_plan.md` 冻结不改，`experiment_plan.md` 只在真实出入时低频更新。 |
| 2026-07-06 | Governance | Revised | 登记 `physics/` 为 GKP 物理仿真代码库，并初始化 Git 仓库。 | 后续修改 `physics/` 需按 task 记录和风险复核执行。 |
| 2026-07-06 | T0.1.1 | Todo -> In Progress | 开始冻结主对象 scope note。 | 输入为 `experiment_plan.md:57-84`，不修改 `rough_plan.md` 和 `experiment_plan.md`。 |
| 2026-07-06 | T0.1.1 | In Progress -> Done | 完成 `docs/tasks/T0.1.1_scope_freeze.md` 并复核 `docs/risks.md`。 | `R-001` 降为 `Mitigated / Medium / Monitor`；当前推荐任务更新为 `T0.1.2`。 |
| 2026-07-06 | T0.1.2 | Todo -> In Progress | 开始冻结平台抽象和 cycle diagram。 | 输入为 `experiment_plan.md:86-90`，参考 `rough_plan.md:694-733`，不修改两份计划文档。 |
| 2026-07-06 | T0.1.2 | In Progress -> Done | 完成 `docs/tasks/T0.1.2_platform_abstraction.md` 并复核 `docs/risks.md`。 | `R-004` 降为 `Mitigated / Low / Monitor`；当前推荐任务更新为 `T0.1.3`。 |
| 2026-07-06 | T0.1.3 | Todo -> In Progress | 开始冻结 CNN 慢回路与 FPGA 快回路接口表。 | 输入为 `experiment_plan.md:92-124`，参考 `rough_plan.md:483-566`，不修改两份计划文档。 |
| 2026-07-06 | T0.1.3 | In Progress -> Done | 完成 `docs/tasks/T0.1.3_dual_loop_interface.md` 并复核 `docs/risks.md`。 | 新增 `R-007` 接口漂移风险；当前推荐任务更新为 `T0.2.1`。 |
| 2026-07-06 | T0.2.1 | Todo -> In Progress | 开始建立 Zotero/文献矩阵并检查本地 Zotero 覆盖。 | 输入为 `experiment_plan.md:130-139` 和 `experiment_plan.md:1245-1342`；Zotero local API 可用。 |
| 2026-07-06 | T0.2.1 | In Progress -> Done | 完成 `docs/literature_matrix.md` 和 `docs/tasks/T0.2.1_literature_matrix.md`，并复核 `docs/risks.md`。 | 新增 `R-008` 和建议插入任务 `T-RISK-20260706-01`；若用户确认暂不补库，则进入 `T0.2.2`。 |
| 2026-07-06 | T-RISK-20260706-01 | Proposed -> In Progress | 开始补齐 High 优先级 Zotero 文献缺口。 | 当前 Zotero 选中目标为 `我的文库 / 量子计算 / 量子纠错 / GKP码`；本任务放在 `T0.2.2` 前。 |
| 2026-07-06 | T-RISK-20260706-01 | In Progress -> Done | 完成 `docs/tasks/T-RISK-20260706-01_zotero_supplement.md`，并将两批共 41 个新增 Zotero key 回填到 `docs/literature_matrix.md`。 | `R-008` 降为 `Mitigated / Low / Monitor`；计划池实际论文/报告已补齐，当前推荐任务保持 `T0.2.2`。 |
| 2026-07-06 | T0.2.2 | Todo -> In Progress | 开始形成 Introduction 可用的 gap statement。 | 输入为 `experiment_plan.md:141-153` 和 `docs/literature_matrix.md`；不修改 `rough_plan.md` 和 `experiment_plan.md`。 |
| 2026-07-06 | T0.2.2 | In Progress -> Done | 完成 `docs/gap_statement.md` 和 `docs/tasks/T0.2.2_gap_statement.md`，并复核 `docs/risks.md`。 | 新增 `R-009` 监控后续 Introduction 扩写时的 novelty claim 外溢；当前推荐任务更新为 `T1.1.1`。 |
| 2026-07-06 | M1.1 | Analysis | 完成 `docs/tasks/M1.1_physics_readiness_analysis.md`，对照 `physics/` 判断 T1.1.1-T1.1.3 是否需要改库。 | 结论：需要新增窄范围 ideal syndrome-level decoder；不插入新 task，缺口由 T1.1.1-T1.1.3 覆盖；新增 `R-010`。 |
| 2026-07-06 | T1.1.1 | Todo -> In Progress | 开始实现理想 1D standard-binning 与高斯翻转概率。 | 输入为 `experiment_plan.md:157-172`、`docs/tasks/M1.1_physics_readiness_analysis.md` 和 `physics/`；计划新增窄范围 ideal syndrome-level decoder。 |
| 2026-07-06 | T1.1.1 | In Progress -> Done | 完成 `physics/ideal_gkp_decoder.py`、`tests/test_ideal_gkp_decoder.py` 和 `docs/tasks/T1.1.1_standard_binning.md`。 | `python -m pytest tests\test_ideal_gkp_decoder.py` 通过；`R-010` 部分缓解，新增 `R-011` 单位约定风险；当前推荐任务更新为 `T1.1.2`。 |
| 2026-07-06 | T1.1.2 | Todo -> In Progress | 开始实现 1D periodic Gaussian MAP hard/soft likelihood。 | 输入为 `experiment_plan.md:174-206` 和 `physics/ideal_gkp_decoder.py`；默认使用 `GKP_SYNDROME_PERIOD = sqrt(pi)`。 |
| 2026-07-06 | T1.1.2 | In Progress -> Done | 完成 `physics/ideal_gkp_decoder.py` 的 1D MAP/LLR 接口和 `docs/tasks/T1.1.2_map_likelihood.md`。 | `python -m pytest tests\test_ideal_gkp_decoder.py` 通过，8 个测试覆盖 standard、MAP hard、MAP soft；`R-010` 剩余范围收缩到 T1.1.3。 |
| 2026-07-06 | T1.1.3 | Todo -> In Progress | 开始实现 2D correlated Gaussian MAP。 | 输入为 `experiment_plan.md:208-230` 和 `physics/ideal_gkp_decoder.py`；默认使用 `GKP_SYNDROME_PERIOD = sqrt(pi)`。 |
| 2026-07-06 | T1.1.3 | In Progress -> Done | 完成 `physics/ideal_gkp_decoder.py` 的 2D correlated MAP 接口和 `docs/tasks/T1.1.3_2d_correlated_map.md`。 | `python -m pytest tests\test_ideal_gkp_decoder.py` 通过，13 个测试覆盖 1D/2D MAP；`R-010` 降为 `Mitigated / Low / Monitor`；当前推荐任务更新为 `T1.2.1`。 |
| 2026-07-06 | M1.2 | Analysis | 完成 `docs/tasks/M1.2_physics_readiness_analysis.md`，对照 `physics/` 判断 T1.2.1-T1.2.3 是否可直接复用现有库。 | 结论：当前库只能部分复用；完整执行 M1.2 需要补 finite-energy 态族输出、逻辑通道/PTM/fidelity 评估和趋势复现 harness；新增 `R-012`，当前推荐任务仍为 `T1.2.1`。 |
| 2026-07-06 | T1.2.1 | Todo -> In Progress | 开始实现 finite-energy approximate GKP 态族、wavefunction、Wigner 和 syndrome distribution。 | 输入为 `experiment_plan.md:236-245`、`docs/tasks/M1.2_physics_readiness_analysis.md` 和 `physics/gkp_state.py`；本任务不实现 T1.2.2 的逻辑通道指标。 |
| 2026-07-06 | T1.2.1 | In Progress -> Done | 完成 `physics/finite_energy_gkp.py`、`tests/test_finite_energy_gkp.py` 和 `docs/tasks/T1.2.1_finite_energy_state_family.md`。 | `python -m pytest` 通过，18 个测试覆盖 ideal decoder 与 finite-energy GKP 态族；`R-012` 部分缓解，当前推荐任务更新为 `T1.2.2`。 |
| 2026-07-06 | T1.2.2 | Todo -> In Progress | 开始实现 effective logical channel / Pauli-twirled finite-energy-aware decoding 指标。 | 输入为 `experiment_plan.md:247-264`、`docs/tasks/M1.2_physics_readiness_analysis.md`、`physics/finite_energy_gkp.py` 和 `physics/ideal_gkp_decoder.py`；不提前声明 Fock-space tomography。 |
| 2026-07-06 | T1.2.2 | In Progress -> Done | 完成 `physics/logical_channel.py`、`tests/test_logical_channel.py` 和 `docs/tasks/T1.2.2_finite_energy_logical_channel.md`。 | `python -m pytest` 通过，23 个测试覆盖 ideal decoder、finite-energy 态族和 logical channel 指标；`R-012` 降为 `Low / Monitor`；当前推荐任务更新为 `T1.2.3`。 |
| 2026-07-06 | T1.2.3 | Todo -> In Progress | 开始实现 finite-energy / optimized decoder 文献趋势复现 harness。 | 输入为 `experiment_plan.md:266-276`、`docs/tasks/M1.2_physics_readiness_analysis.md`、`physics/finite_energy_gkp.py`、`physics/logical_channel.py` 和 `physics/ideal_gkp_decoder.py`；优先复现 finite-energy standard binning 次优及优化优势随能量升高收缩的趋势。 |
| 2026-07-06 | T1.2.3 | In Progress -> Done | 完成 `physics/finite_energy_trends.py`、`tests/test_finite_energy_trends.py` 和 `docs/tasks/T1.2.3_finite_energy_trend_reproduction.md`。 | `python -m pytest` 通过，30 个测试覆盖 ideal decoder、finite-energy 态族、logical channel 指标和 finite-energy 趋势复现；`R-012` 剩余监控点收缩到 Phase 2.3 / M5.2；当前推荐任务更新为 `T1.3.1`。 |
| 2026-07-06 | T1.3.1 | Todo -> In Progress | 开始实现 synthetic drift process generator。 | 输入为 `experiment_plan.md:282-305`、`physics/logical_tracking.py`、`physics/error_correction.py`、`physics/noise_channels.py` 和既有 `run_with_drift` 回调约定；优先复用现有 `sigma/delta/theta/error_bias` 语义并补齐 mean、variance、loss、outlier-rate、step、telegraph、burst 漂移状态。 |
| 2026-07-06 | T1.3.1 | In Progress -> Done | 完成 `physics/drift_processes.py`、`tests/test_drift_processes.py` 和 `docs/tasks/T1.3.1_drift_process_generator.md`。 | `python -m pytest` 通过，36 个测试覆盖 drift process、ideal decoder、finite-energy 态族、logical channel 和 finite-energy 趋势；新增 `R-013` 监控后续是否完整消费 `mu/eta/p_outlier/burst`；当前推荐任务更新为 `T1.3.2`。 |
| 2026-07-06 | T1.3.2 | Todo -> In Progress | 开始实现知道真实 `DriftState` 的 oracle MAP。 | 输入为 `experiment_plan.md:307-317`、`physics/drift_processes.py` 和 `physics/ideal_gkp_decoder.py`；优先复用 2D periodic Gaussian MAP，并把 oracle 明确限定为不可部署上界。 |
| 2026-07-06 | T1.3.2 | In Progress -> Done | 完成 `physics/oracle_map.py`、`tests/test_oracle_map.py` 和 `docs/tasks/T1.3.2_oracle_map.md`。 | `python -m pytest` 通过，42 个测试覆盖 oracle MAP、drift process、ideal decoder、finite-energy 态族、logical channel 和 finite-energy 趋势；`R-002` 进一步缓解，`R-013` 部分缓解；当前推荐任务更新为 `T1.3.3`。 |
| 2026-07-12 | T-RISK-20260712-01 | Proposed -> In Progress | 开始按三篇 GKP 实验/综述材料重构任务板。 | 保留已完成任务与当前指针；目标是建立 sBs-first 数字孪生、混合状态噪声、三时间尺度控制、先 FPGA 后论文的执行链。 |
| 2026-07-12 | T-RISK-20260712-01 | In Progress -> Done | 完成 v2 任务板、实验计划低频修订、任务记录和风险同步。 | 当前推荐任务保持 `T1.3.3`；`docs/rough_plan.md` 未修改；后续从 M1.4 起执行新顺序。 |
| 2026-07-13 | T-RISK-20260713-01 | Proposed -> In Progress | 开始按 NMF PRL 正文、补充材料和图片定向审计 v2 任务板。 | 聚焦 controller/decoder 边界、memory 因果证据、Feedback-GRAPE 可行性和约 300 元 FPGA 的可部署降级路径。 |
| 2026-07-13 | T-RISK-20260713-01 | In Progress -> Done | 完成 v2.1 任务板、实验计划第 15 节、任务记录和 R-019—R-023 同步。 | 新增 teacher/student 与双 oracle 证据链；失败时回退 v2 主线；当前推荐任务保持 `T1.3.3`。 |
| 2026-07-13 | T-RISK-20260713-02 | Proposed -> In Progress | 开始按六篇补充论文对 v2.1 做小幅证据补强。 | 不重排阶段；只补中保真度模型、通道上界、top-K baseline 和 secondary protocol 边界。 |
| 2026-07-13 | T-RISK-20260713-02 | In Progress -> Done | 完成 v2.2 任务板、实验计划第 16 节、任务记录和 R-024—R-026 同步。 | 新增 3 个正常执行任务并增强既有 evidence gate；当前推荐任务保持 `T1.3.3`。 |
| 2026-07-14 | T0.1.1 | Todo -> In Progress | 重新审计当前仓库的 `physics/`、论文边界材料与既有测试，而不是沿用与主表冲突的旧完成日志。 | 新任务序列以当前仓库产物和当前验证为准。 |
| 2026-07-14 | T0.1.1 | In Progress -> Done | 完成 `docs/tasks/T0.1.1_scope_freeze.md`、`docs/new_tasks/T0.1.1_scope_freeze.md` 和 `docs/new_risks.md`；本地 conda smoke 跑通 Wigner、modular syndrome、linear correction 与 20-round `full_qec` tracking。 | 明确只冻结 single-mode square approximate GKP repeated-memory 主对象；未把 smoke 冒充物理验证；根级 pytest 收集、单位约定和 direct physics tests 缺口已登记；当前推荐任务更新为 `T0.1.2`。 |
| 2026-07-14 | T0.1.2 | Todo -> In Progress | 开始按本地三份核心论文、现有 `physics/`/`cnn_fpga/` 和硬件 claim 边界冻结平台抽象。 | 目标是可实现的 stage contract，不把 cavity/transmon 文献映射写成当前实验事实。 |
| 2026-07-14 | T0.1.2 | In Progress -> Done | 完成 `docs/tasks/T0.1.2_platform_abstraction.md` 和 `docs/new_tasks/T0.1.2_platform_abstraction.md`；cycle diagram、阶段 I/O、失败分支和 platform-independent 数学 contract 完整；本地 conda 单轮 QEC software-proxy smoke 通过。 | 明确 ECD/sBs、IQ/ADC、leakage/reset 状态机尚未实现，`4.924 us` 仅是文献时序；当前推荐任务更新为 `T0.1.3`。 |
| 2026-07-14 | T0.1.3 | Todo -> In Progress | 开始审计现有 `ParamBank`、scheduler、fast/slow runtime、fixed-point 与 HIL config，冻结双回路真实接口。 | 优先复用现有代码，同时区分 current representation 和 target wire contract。 |
| 2026-07-14 | T0.1.3 | In Progress -> Done | 完成 `docs/tasks/T0.1.3_dual_loop_interface.md` 和 `docs/new_tasks/T0.1.3_dual_loop_interface.md`；接口覆盖输入/输出/频率/位宽/atomic commit/fallback；focused runtime tests `4 passed`，direct stage/commit 与 Q4.20 smoke 通过。 | CRC/stale/deadline fallback 与 raw-count histogram adapter 尚未实现，已登记 R-N007/R-N008；当前推荐任务更新为 `T0.2.1`。 |
| 2026-07-14 | T0.2.1 | Todo -> In Progress | 开始对 38 条 Zotero 快照、live Zotero API、local papers 与 2025–2026 一级来源做四线覆盖和去重审计。 | 使用 `nature-academic-search` 多源/去重流程和 Zotero 只读查询；并行审计 GKP、adaptive、NN/FPGA 三条子线。 |
| 2026-07-14 | T0.2.1 | In Progress -> Done | 完成 `docs/literature_matrix.md`、`docs/tasks/T0.2.1_literature_matrix.md` 和 `docs/new_tasks/T0.2.1_literature_matrix.md`；40 篇逐行字段完整，finite-energy/hardware/Zotero 分级，DOI/arXiv 去重检查通过。 | 真实 QPU闭环与 FPGA/kernel/software latency 已分层；4 个 Zotero 缺口和 preprint metadata drift 登记为 R-N009/R-N010；当前推荐任务更新为 `T0.2.2`。 |
| 2026-07-14 | T0.2.2 | Todo -> In Progress | 开始基于四线文献矩阵写 gap statement 和英文 Introduction 草稿。 | 必须把 direct baselines、oracle 上界、finite-energy-aware、hardware-aware simulation 和证据降级边界写成可验证 contract。 |
| 2026-07-14 | T0.2.2 | In Progress -> Done | 完成 `docs/gap_statement.md`、`docs/tasks/T0.2.2_gap_statement.md` 和 `docs/new_tasks/T0.2.2_gap_statement.md`；形成中文 gap、五段英文 Introduction、术语表、7 个可证伪 RQ、baseline/oracle/bound contract 和 claim ladder。 | 30 个 canonical citation keys 全部存在；novelty 只写联合审计缺口，R-N011 监控绝对首次外溢；当前推荐任务更新为 `T1.1.1`。 |
| 2026-07-14 | T1.1.1 | Todo -> In Progress | 开始冻结单轴 square-GKP spacing、syndrome period 和 standard-binning 判决，并推导 Gaussian displacement logical-flip 概率。 | 先对齐现有 `physics/` convention，再用解析区间和独立 Monte Carlo 做数值验证。 |
| 2026-07-14 | T1.1.1 | In Progress -> Done | 完成 `physics/ideal_gkp_decoder.py`、`tests/test_ideal_gkp_decoder.py`、任务文档和新任务完成记录；实现 cell-parity standard binning、erfc interval sum 与 Fourier probability。 | focused `12 passed`；40万样本 MC 与解析值相差 `1.18e-4`（0.23 SE）；tests/ 为 `78 passed,4 failed`，旧缺失 FR8/P4 文档已登记 R-N012；当前推荐任务更新为 `T1.1.2`。 |
| 2026-07-14 | T1.1.2 | Todo -> In Progress | 开始实现 centered-syndrome 下的 even/odd periodic Gaussian likelihood、LLR、MAP hard/soft 和统一三模式入口。 | 要求数值稳定、定义 tie/符号 convention，并与 brute-force alias sum 和 posterior normalization 对照。 |
| 2026-07-14 | T1.1.2 | In Progress -> Done | 完成 1D periodic-Gaussian coset likelihood、稳定 LLR、hard/soft MAP、三模式入口、任务文档和完成记录；修复超大 lattice index 静默溢出分支。 | focused `24 passed`、邻近回归 `28 passed`；显式 tests/ 为 `90 passed,4 failed`，仍仅是 R-N012 两份旧文档缺失；当前推荐任务更新为 `T1.1.3`。 |
| 2026-07-14 | T1.1.3 | Todo -> In Progress | 开始实现二维相关 Gaussian 的四 logical coset likelihood、joint MAP 与 independent-axis 对照。 | 必须验证 `rho=0` 严格退化、协方差失败分支，并用配对 Monte Carlo 证明强相关场景的 joint decision gain 不是手选样例。 |
| 2026-07-14 | T1.1.3 | In Progress -> Done | 完成 covariance validator、四 logical-coset joint MAP、independent-axis 对照、复杂度安全门、任务文档和完成记录。 | focused `34 passed`、邻近 `38 passed`；强相关配对 MC 绝对改善 `0.07817`、`z=29.60`；显式 tests/ 为 `100 passed,4 failed`（R-N012）；当前推荐任务更新为 `T1.2.1`。 |
| 2026-07-14 | T1.2.1 | Todo -> In Progress | 开始审计现有 `gkp_state.py` 是否真实支持 Gaussian peaks+envelope 与 damped-projector 两态族、四逻辑态、wavefunction/Wigner-like/syndrome 输出。 | 先判断既有 analytical Wigner 是否只是可视化代理，再建立有归一化、收敛和极限趋势验证的 finite-energy state-family API。 |
| 2026-07-14 | T1.2.1 | In Progress -> Done | 完成 normalized Gaussian-envelope 与 `exp(-Delta^2 n)` damped-projector 四态 family、syndrome folding、sampled Wigner、任务文档和完成记录；legacy fallback 明确降级为 heuristic visualization。 | focused `14 passed`、邻近组合 `52 passed`；显式 tests/ `114 passed,4 failed`（R-N012）；新增 R-N013 路由风险；当前推荐任务更新为 `T1.2.2`。 |
| 2026-07-14 | T1.2.2 | Todo -> In Progress | 开始定义 parity-output decoder 的 Pauli logical channel，而不是把 half-cell crossing probability 直接等同于完整 finite-energy channel。 | 必须明确 q/p parity 到 X/Z/Y 的映射、CPTP/Pauli-twirl 边界、PTM 与 `F_e/F_avg` 恒等式，并对 correlated joint posterior 与 Monte Carlo channel estimate 做交叉验证。 |
| 2026-07-14 | T1.2.2 | In Progress -> Done | 完成 arbitrary parity decoder 的 XOR residual -> Pauli channel、PTM/fidelity、finite-state/noise alias response、解析 convolution、任务文档和完成记录。 | focused `13 passed`、组合 `65 passed`；独立 12 万样本 MC 差 `1.74 SE`；全量 `127 passed,4 failed`（R-N012）；新增 R-N014 Pauli-twirl 边界；当前推荐任务更新为 `T1.2.3`。 |
| 2026-07-14 | T1.2.3 | Todo -> In Progress | 开始建立可复现 finite-energy decoder 趋势 harness，比较 gain=1 standard correction 与 state/noise-aware shrinkage。 | 必须先给出 correction action 和 loss functional，扫描多个 `Delta`/noise 点并报告 paired confidence；若趋势只在 effective model 成立，文档不得升级为完整物理 recovery 结论。 |
| 2026-07-14 | T1.2.3 | In Progress -> Done | 完成 additive finite-energy syndrome-noise shrinkage harness、独立 train/held-out eval、五点趋势、paired CI/McNemar、任务文档和完成记录。 | focused `9 passed`、组合 `74 passed`、全量 `136 passed,4 failed`（R-N012）；新增 R-N015 fidelity 边界；当前推荐任务更新为 `T1.3.1`。 |
| 2026-07-14 | T1.3.1 | Todo -> In Progress | 开始审计 `noise_channels.py`、`error_correction.run_with_drift` 与现有 config 字段，定义统一 `DriftState` 和七类 synthetic drift processes。 | 要求每个过程有 deterministic seed、参数边界、时间索引语义、truth provenance 与 callback adapter；burst/telegraph 不得简化成无状态随机点。 |
| 2026-07-14 | T1.3.1 | In Progress -> Done | 完成 full `DriftState`、七类 drift、mixture sampler、旧回调适配、17 项 direct tests、统计复核、任务文档与风险同步。 | focused `17 passed`、邻近 `87 passed`、全量 `153 passed,4 failed`（R-N012）；新增 R-N016/R-N017；当前推荐任务更新为 T1.3.2。 |
| 2026-07-14 | T1.3.2 | Todo -> In Progress | 开始定义直接消费 full `DriftState` 的 mean/covariance/mixture-aware periodic oracle MAP。 | 禁止走 legacy scalar adapter；先冻结 parity/action、alias truncation、mixture log-sum-exp、不可部署上界和独立穷举/MC 验证。 |
| 2026-07-14 | T1.3.2 | In Progress -> Done | 完成 full-state periodic mixture oracle、trajectory、显式 loss policy、9 项 direct tests、独立 alias 穷举、MC posterior-risk calibration、任务文档和风险同步。 | focused `9 passed`、邻近 `96 passed`、全量 `162 passed,4 failed`（R-N012）；新增 R-N018，R-N016 降低迫切度；当前推荐任务更新为 T1.3.3。 |
| 2026-07-14 | T1.3.3 | Todo -> In Progress | 开始冻结 static/dual/oracle 三者同一 paired sample set 上的 regret、oracle-gap、分母退化和置信区间语义。 | 必须区分 absolute gap、fraction closed、out-of-bracket/zero-denominator，不得用截断掩盖 dual 差于 static 或“超过”有限样本 oracle。 |
| 2026-07-14 | T1.3.3 | In Progress -> Done | 完成 raw/ratio gap、paired difference、McNemar、八类联合 outcome bootstrap、denominator/reliability gate、10 项 direct tests、任务文档和风险同步。 | focused `10 passed`、邻近 `106 passed`、全量 `172 passed,4 failed`（R-N012）；新增 R-N019；当前推荐任务更新为 T1.3.4。 |
| 2026-07-14 | T1.3.4 | Todo -> In Progress | 开始审计现有 window/EKF/UKF/particle-filter baseline 与 T1.3 full-state truth/metric contract，建立同 trace synthetic alignment benchmark。 | 先复用真实 baseline API；必须同时有 static、oracle 和至少一个 deployable adaptive estimator，使用 paired decisions，并在继续 CNN 前证明非退化可利用 gap。 |
| 2026-07-14 | T1.3.4 | In Progress -> Done | 完成真实 Window/EKF API、独立 calibration、one-window delay、逐窗 trace hash、paired oracle-gap gate、18 项 direct tests、任务文档和风险同步。 | 72k samples 上 EKF 关闭 `72.14%` gap，95% bootstrap CI `[70.44%,73.78%]`；全量 `190 passed,4 failed`（R-N012）；新增 R-N020；当前推荐任务更新为 T1.4.1。 |
| 2026-07-14 | T1.4.1 | Todo -> In Progress | 开始冻结 simulation、synthesis estimate、约 300 元真实板 measurement、HIL/replay 和真实量子实验五层 claim ladder。 | 先审计已有 evidence/claim 文档与真实产物；每项 claim 必须给出 allowed/forbidden wording、升级门和失败降级路径。 |
| 2026-07-14 | T1.4.1 | In Progress -> Done | 完成五层 claim ladder、8 条 claim registry、2 条正交 evidence lane、机器 JSON、8 项 artifact-backed tests、任务文档和风险同步。 | 当前最高仅 CL1；T48 host runtime 与 T72 real-board NO_GO 分离；全量 `198 passed,4 failed`（R-N012）；新增 R-N021；当前推荐任务更新为 T1.4.2。 |
| 2026-07-14 | T1.4.2 | Todo -> In Progress | 开始对约 300 元 FPGA 做实际型号/器件、I/O、时钟、存储、资源和可测量边界的证据审计。 | 必须区分已选定实物、候选板、datasheet 规格、综合可用资源和本项目实测；没有实物/bitstream 时不得虚构“实际板卡”。 |
| 2026-07-14 | T1.4.2 | In Progress -> Done | 完成 Tang Nano 20K reference-target、厂商资源、5 类接口、容量/传输算术、现有后端不兼容和数字测量边界的 Markdown/JSON contract 与 9 项 tests。 | Focused 9 passed、相邻 17 passed、全量 207 passed/4 failed（R-N012）；实物/最新报价/吞吐/综合/板测均 fail closed；新增 R-N022；T1.4.3 转 In Progress。 |
| 2026-07-14 | T1.4.3 | In Progress -> Done | 完成 CD1/CD2、TS1/TS2/TS3、XIF01--04、AP01--09、FB01--14 的 Markdown/JSON contract 和 11 项 direct tests。 | Focused 11 passed、相邻 28 passed、全量 218 passed/4 failed（R-N012）；hidden-truth、record-only deadline 与 atomicity 缺口未伪装完成；新增 R-N023；T1.4.4 转 In Progress。 |
| 2026-07-14 | T1.4.4 | In Progress -> Done | 完成 8-source/51-entry paper-parameter registry、Markdown/JSON 同步和 14 项 direct tests。 | DOI-first formal 去重；Sivak 双 cycle 与 1546/1548 ns 差异、Puviani 理想化/best-of-20、secondary/pending gates 均 fail closed；全量 232 passed/4 failed（R-N012）；T1.4.5 转 In Progress。 |
| 2026-07-14 | T1.4.5 | In Progress -> Done | 完成 11-role decoder/controller terminology registry、artifact/legacy mapping 和 14 项 direct tests。 | 三种上界、teacher/student、host/fast path、`oracle_delayed`/`teacher_mode` 均不可混称；全量 246 passed/4 failed（R-N012）；T2.0.1 转 In Progress。 |
| 2026-07-14 | T2.0.1 | In Progress -> Done | 完成 4-protocol hierarchy、14 条 source anchors、6 条 nonmixing rules、4 个 promotion gates 和 12 项 direct tests。 | sBs constituent/full cycle、paired outcome 与 `f` reset branch 分离；sharpen--trim timing fail closed；全量 258 passed/4 failed（R-N012）；T2.0.2 转 In Progress。 |
| 2026-07-14 | T2.0.2 | In Progress -> Done | 完成 grouped CPTP sBs error-space instrument、Pauli frame、quantum/population/trajectory APIs 和 21 项 direct tests。 | completeness/branch Choi/C0 no-error/多 `C_i` 同层/随机 density/Monte Carlo/负路径通过；全量 279 passed/4 failed（R-N012）；T2.0.3 转 In Progress。 |
| 2026-07-14 | T2.0.3 | In Progress -> Done | 完成 ideal/hidden/observed/reset 四层模型、full confusion、f/higher leakage、e/leakage runs 和 20 项 direct tests。 | label Z/X 与执行 X/Z、schema isolation、confusion/reset/streak Monte Carlo 和负路径通过；全量 299 passed/4 failed（R-N012）；T2.0.4 转 In Progress。 |
| 2026-07-14 | T2.0.4 | In Progress -> Done | 完成 18-phase Table S3 constituent FSM、连续 X→Z full cycle、observed/reset/VR 接线、Pauli frame 和 13 项 direct tests。 | `1546/1548 ns`、`2332/2380 ns` scope 差异与 source anchor 保留；全量 312 passed/4 failed（R-N012）；新增 R-N028，T2.0.5 转 In Progress。 |
| 2026-07-14 | T2.0.5 | In Progress -> Done | 完成 nearest-logical-operation distance、depth injection、T2.0.2 transition/T2.0.3 observation 接线、bootstrap/tolerance/failure diagnostics 和 26 项 direct tests。 | 4096-shot 中点 e-run `4.883 [4.846,4.919]`、双边 Spearman `1/-1`、未受影响象限负控通过；全量 338 passed/4 failed（R-N012）；新增 R-N029，T2.0.6 转 In Progress。 |
| 2026-07-14 | T2.0.6 | In Progress -> Done | 完成 shared hidden/observed trajectory、all-gg occupancy estimator、leakage-run removal、long-lag paired bootstrap 和 29 项 direct tests。 | hidden/syndrome `0.813565/0.813524`；tail paired shrink CI `[0.001684,0.005058]`，no-higher ablation 触发失败；全量 367 passed/4 failed（R-N012）；新增 R-N030，T2.1.1 转 In Progress。 |
| 2026-07-14 | T2.1.1 | In Progress -> Done | 完成 full `DriftState` mixed-state stream、observed/truth schema、loss/mixture/modular/logical、g/e/leakage 与 recovery-depth 因果状态，共 21 项 direct tests。 | 20k mixture、8k loss variance、5k confusion、leakage-ended recovery、single-axis correction、seed/prefix 均通过；全量 390 passed/4 failed（R-N012）；新增 R-N031，T2.1.2 转 In Progress。 |
| 2026-07-14 | T2.1.2 | In Progress -> Done | 完成 observed-only multi-round memory：nearest-lift residual、actual correction、confidence、Pauli/phase frame、runs、bank version 和 deadline state，共 26 项 direct tests。 | correction sign 与旧 fast loop 对齐，truth-step rejection、transactional failures、real ParamBank commit 与 segmented replay 通过；全量 418 passed/4 failed（R-N012）；新增 R-N032，T2.1.3 转 In Progress。 |
| 2026-07-14 | T2.1.3 | In Progress -> Done | 完成 trajectory 向量化 million-cycle core、q/p/any/trajectory 指标、target-weighted burst/leakage strata、trajectory-cluster CI、zero-event bound、CLI/JSON 和 protocol hierarchy 同步。 | 解析 Gaussian、recovery 因果消融、known-mixture/allocation 负测均通过；production 1e6 cycles 为 `0.2531 s`、约 `3.95e6 host cycles/s`；全量 439 passed/4 failed（R-N012）；R-N031 降为 Mitigated，新增 R-N033，T2.2.1 转 In Progress。 |
| 2026-07-14 | T2.2.1 | In Progress -> Done | 完成 physical/observed 两 lane 的 channel/data-GKP/ancilla-GKP/measurement/envelope 分解、Mehler lattice envelope、analytic budget、wrapped correction、6 点 high-squeezing sweep、CLI/JSON 与 protocol hierarchy 同步。 | state-family variance、26 万逐项 covariance、50 万 analytic LER、envelope ablation、exact Delta-zero endpoint 均通过；production 150 万 samples，最大 covariance relative error `0.004554`；全量 466 passed/4 failed（R-N012）；R-N015 降为 Mitigated，新增 R-N034，T2.2.2 转 In Progress。 |
| 2026-07-14 | T2.2.2 | In Progress -> Done | 完成 sBs constituent×stage ancilla bit/phase/readout overlay、既有 full reset/leakage kernel 接线，以及独立 sharpen--trim 四轮 `+y/-y` state machine、3×2 confusion、hidden carry、secondary non-executable registry、CLI/JSON 与 protocol hierarchy。 | 首轮审计修复 deployable hidden-counter 泄漏和 stochastic fault 写入 controller frame；direct 27、adjacent 115、production 80k+80k 全门通过；全量 494 passed/4 failed（R-N012）；R-N025 降为 Mitigated，新增 R-N035，T2.2.3 转 In Progress。 |
| 2026-07-14 | T2.2.3 | In Progress -> Done | 完成 request→AWG/DAC→pulse/latency/virtual-rotation→physical residual 三层因果模型、两种 noncommuting order、exact moments、vectorized batch/trajectory、Q4.20 integration、CLI/JSON 与 protocol hierarchy。 | direct 33、adjacent 110、2×80k moments、100k production、6/8/10/12-bit 和 0/2/5/10-us sweeps 全通过；全量 528 passed/4 failed（R-N012）；新增 R-N036，T2.3.1 转 In Progress。 |
| 2026-07-14 | T2.3.1 | In Progress -> Done | 完成 normalized finite-energy GKP→Hermite/Fock projection、immutable density/diagnostics、displacement、pure loss、sparse thermal Lindblad、phase diffusion、Kerr、modular POVM backaction、high-Fock proxy、CLI/JSON 与 protocol hierarchy。 | direct 33、adjacent 81、12/18/24/30 cutoff 捕获率增至 `0.999996`、10 production gates 全通过；全量 562 passed/4 failed（R-N012）；新增 R-N037，T2.3.2 转 In Progress。 |
| 2026-07-14 | T2.3.2 | In Progress -> Done | 以官方 TeX 核对后的 analytic SBS Kraus 替换 modular-pump 近似，完成 `sqrt(2)` canonical 坐标桥、raw/completed 双轨、idle、hidden/observed/classical action、X→Z frame、logical projection、exact branch、CLI/JSON 与 protocol hierarchy。 | direct 99、adjacent 177、full 629 passed/4 failed（R-N012）；clean conditional/survival `0.999612/0.822991`、photon-error gain `0.423853`、100k MC z `1.7582`、五点 cutoff 与 16 gates 全通过；新增 R-N038，T2.3.3 转 In Progress。 |
| 2026-07-14 | T2.3.3 / T2.3.8 | In Progress -> Blocked / Todo -> In Progress | 审计发现 T2.3.3 的通过标准强制要求 noise-transfer surrogate，但唯一实现任务 T2.3.8 排在其后且仓库无现成实现；将既有 T2.3.8 前移为单一前置依赖，完成后返回 T2.3.3。 | 不插入新 task、不缩减 T2.3.3 lane、不以 `finite_squeezing_noise` 冒充 Heisenberg surrogate；新增 R-N039，任务 ID/范围保持不变。 |
| 2026-07-14 | T2.3.8 | In Progress -> Done | 完成 signal/fluctuation/logical-jump 分离、loss/measurement/gain covariance、exact Gaussian alias、correlated Fréchet-only、dB→Delta、四逻辑态 state/Fock q-domain alignment、CLI/JSON 与 protocol hierarchy。 | 45 direct、194 adjacent、full 675 passed/4 failed（R-N012）、200k MC 与 14 production gates；10/12 dB 对齐，3 dB proxy 偏差 `29.79%` 与四态 spread `42.33%` 作为证否保留；新增 R-N040，T2.3.3 恢复 In Progress。 |
| 2026-07-14 | T2.3.3 | In Progress -> Done | 完成四 lane 共同输入、Fock folded response/SBS native metrics、effective 200k/点、exact noise-transfer、direct syndrome、12 dB 五 cutoff 和四项 error attribution。 | direct 42、adjacent 278、full 718 passed/4 failed（R-N012）、production 14 gates；high-dB noise↔syndrome q-gap `3.93e-6`，3 dB clipping gap `0.016766`；Fourier-p gap `>0.48` 触发 R-N041 与插入任务。 |
| 2026-07-14 | T-RISK-20260714-01 | Todo -> In Progress | 开始核对 commutator-preserving operational/canonical quadrature mapping、Fourier reciprocal lattice 和 q/p numerical roundtrip。 | T2.3.4 暂不启动；two-axis coherent Fock claim 保持 blocked。 |
| 2026-07-14 | T-RISK-20260714-01 | In Progress -> Done | 完成 `quadrature_conventions.py`、32 direct tests、15 machine gates、完整 wavefunction/variance/Fock 回灌、protocol/PC-N01 同步与历史 task 勘误。 | canonical q/p high-dB gap `1.51e-7`；legacy gap `>0.418`；full 751 passed/4 known R-N012 failures；R-N041 Mitigated。 |
| 2026-07-14 | T2.3.4 | Todo -> In Progress | 开始短时域可微 SBS trajectory simulator 的输入/输出、通过标准、失败分支与 differentiability 方案审计。 | 当前推荐任务更新为 `T2.3.4`；不提前启动 T2.3.5。 |
| 2026-07-14 | T2.3.4 | In Progress -> Done | 完成 `differentiable_sbs_trajectory.py`、37 direct、CPU/CUDA 各 17 machine gates、history-policy、branch normalization、resource profile 与双任务记录。 | 四分支和 `1.0`；CPU/CUDA open-loop/history gradient norm `0.9365/1.0424`；R-N042 登记，当前推荐更新为 T2.3.5。 |
| 2026-07-14 | T2.3.5 | Todo -> In Progress | 开始冻结 Feedback-GRAPE reward-path、score-path、baseline/stop-gradient、finite-difference 与 stochastic estimator 验收合同。 | 不把 T2.3.4 graph connectivity 冒充 T2.3.5 数值梯度验证。 |
| 2026-07-14 | T2.3.5 | In Progress -> Done | 完成 `feedback_grape_gradient.py`、exact 4/16 branch、分项 FD、step sweep、12,288 trajectory MC、32 direct 与 15 machine gates。 | decomposition `5.55e-17`；FD `<3.23e-10`；最大 `1.120 SE`；R-N043 登记，当前推荐更新为 T2.3.6。 |
| 2026-07-14 | T2.3.6 | Todo -> In Progress | 开始冻结 cutoff/batch/horizon 网格、warm-up/repeat、CPU/CUDA 内存、OOM/数值稳定和 2--10 cycle teacher envelope 判据。 | 不把 T2.3.4 单点 resource profile 当成 feasibility scan。 |
| 2026-07-14 | T2.3.6 | In Progress -> Done | 完成 72,913 参数 causal GRU 的 65 点 trajectory/backward/Adam 资源扫描、CPU/CUDA 画像、机器数据、source CSV、多格式科研图和双任务记录。 | 63 pass、1 memory exceeded、1 runtime exceeded；cutoff 16/batch 16 的 2--10 cycles 全通过；40 direct；全量 754 passed/3 skipped/4 个已知 R-N012 failures；R-N043 仍保留训练/ranking 边界，当前推荐更新为 T2.3.7。 |
| 2026-07-14 | T2.3.7 | Todo -> In Progress | 开始冻结 standard、memoryless MF、NMF 的同仿真器、同噪声、同物理时间、独立 train/validation/test seeds 与方向性 lifetime ranking 合同。 | 不把未训练策略、训练集末态 fidelity 或不同周期数比较冒充 NMF ranking。 |
| 2026-07-15 | T2.3.7 | In Progress -> Done | 完成 V3 strict-split 5+5 agent Feedback-GRAPE、schema-v3 hash checkpoint、8-seed test、cutoff-16 confirmation、hidden-reset 反事实、source data/科研图和双任务记录。 | 主 cutoff standard/MF/NMF lifetime `2.7477/6.5347/6.7408`，5/5 NMF 高于配对 MF；confirmation 总排序保持但 reset 机制反转；联合 `103 passed`、邻接 `140 passed`、全量 `755 passed/4 skipped/4 个 R-N012 已知 failures`。 |
| 2026-07-15 | T2.4.1 | Todo -> In Progress | 开始审计文献 measurement/ADC/control/AWG 时序与项目 UART/replay/FPGA/action 时序的来源、测量状态、算术关系和禁止混用规则。 | 只建立双 budget 与 provenance contract，不提前实现 T2.4.2 backlog/jitter 或 T2.4.3 fixed-point 模型。 |
| 2026-07-15 | T2.4.1 | In Progress -> Done | 完成 dual-latency-budget-v1、Sivak/Puviani source anchors、项目 cadence/software model/UART capacity、七类未测 null、23-gate validator、SHA-bound snapshot 和双任务记录。 | direct `22 passed`、相邻 `50 passed`、全量 `777 passed/4 skipped/4 个 R-N012 已知 failures`；R-N006/R-N028 降为 Mitigated，当前推荐更新为 T2.4.2。 |
| 2026-07-15 | T2.4.2 | Todo -> In Progress | 开始冻结 backlog、jitter、deadline miss、input burst、transport pause、parameter-update conflict 与 FIFO overflow 的可执行状态机、指标和失败分支。 | 必须量化 LER/availability，且不把 software timing model 冒充实板计时；不提前进入 T2.4.3 fixed-point/LUT。 |
| 2026-07-15 | T2.4.2 | In Progress -> Done | 完成 scheduler backlog/pause/window-age、ParamBank second-writer rejection、7 场景 8-seed paired stress、56-row Source Data、13 machine gates 与双任务记录。 | 3,584,000 ticks；combined LER 增量 `0.27957 [0.27645,0.28250]`、availability 下降 `0.88480 [0.88049,0.88842]`；67 direct+adjacent、全量 785 passed/4 skipped/4 个已知 R-N012 failures；R-N045 登记，当前推荐更新为 T2.4.3。 |
| 2026-07-15 | T2.4.3 | Todo -> In Progress | 开始审计 ADC/replay input、LUT、LLR、threshold、state estimate、parameter update granularity 的现有位宽/量化路径、资源代理和失败分支。 | 必须生成 precision-resource-LER 曲线并覆盖 bank error；不把 host fixed-point simulation 冒充 synthesis/board result。 |
| 2026-07-15 | T2.4.3 | In Progress -> Done | 完成 integer ADC/state/LLR/threshold/LUT/double-bank、42 profiles、4 bank faults、368 paired runs、Source Data 与 Python-only 四格式科研图。 | 11/11 model gates、5/5 figure gates、63 direct+adjacent passed；全量 `796 passed,4 skipped,4 个 R-N012 已知 failures`；high precision 收敛 float，joint 非单调与 stale/torn CI 跨零保留；R-N046 登记，当前推荐更新为 T3.1.1。 |
| 2026-07-15 | T3.1.1 | Todo -> In Progress | 开始审计 fixed half-cell nearest-lattice recovery 是否已进入所有主要 comparison schema、runner、报告和 provenance，并冻结独立 baseline 合同。 | 复用已验证 standard-binning math，但不得把“函数存在”冒充“所有主要比较已接入”。 |
| 2026-07-15 | T3.1.1 | In Progress -> Done | 完成 observed-only standard decision、hidden-truth paired evaluator、major-comparison registry、T1.3.4 五行接入、production JSON 和双任务记录。 | 72k paired trace 上 standard/static LER `0.060417/0.061389`，差 `-0.000972 [-0.001280,-0.000664]`；10 gates、92 focused+adjacent、全量 `817 passed/4 skipped/4 个 R-N012 已知 failures`；R-N047 登记，当前推荐更新为 T3.1.2。 |
| 2026-07-15 | T3.1.2 | Todo -> In Progress | 开始冻结与 evaluation 隔离的 training-set average noise/observation parameter contract，并审计旧 static-calibration row 为何略差于 standard。 | 必须在同 trace 保留 standard 行和 paired counterevidence，不得调参到 evaluation 或删除简单基线。 |
| 2026-07-15 | T3.1.2 | In Progress -> Done | 完成 training-state total-covariance fit、frozen static decoder、active/future schema 接入、8-seed 576k Source Data 和双任务记录。 | standard/static/oracle `0.058870/0.024498/0.011340`，gain `0.034372 [0.033798,0.034946]`；9 gates、119 focused+adjacent、全量 `844 passed/4 skipped/4 个 R-N012 已知 failures`；旧 EKF primary gate 降为 false，R-N047 Mitigated/R-N048 登记，当前推荐更新为 T3.1.3。 |
| 2026-07-15 | T3.1.3 | Todo -> In Progress | 开始审计 T1.3.2 full-state oracle 是否直接接入所有 required decoder schemas，并扩展 regime/leakage state 的上界与 truth-leakage/provenance gate。 | 必须同时保留 standard/static，oracle 只作不可部署 assumed-model upper reference，不能进入 deployable action path。 |
| 2026-07-15 | T3.1.3 | In Progress -> Done | 完成 full-state/regime oracle schema、truth-only leakage flag、4-regime 320k matrix、8k protocol envelope、Source Data 和双任务记录。 | static/oracle `0.076419/0.025103`，gap `0.051316 [0.050474,0.052157]`；1,616 leakage 仅 flag，cost `[0.159,0.361]`；10 gates、99 focused+adjacent、全量 `858 passed/4 skipped/4 个 R-N012 已知 failures`；术语源码锚同步，R-N049 登记，当前推荐更新为 T3.1.4。 |
| 2026-07-15 | T3.1.4 | Todo -> In Progress | 开始审计 finite-energy state-family、sBs observation/reset、effective logical channel 与现有 optimized shrinkage，冻结一个不消费 hidden truth 的 static protocol-aware decoder。 | 必须明确 observation likelihood/action/loss、相对 standard/static/oracle 的位置和适用 fidelity；不得把 protocol heuristic 冒充 full recovery optimum。 |
| 2026-07-15 | T3.1.4 | In Progress -> Done | 完成 exact sBs observation/reset likelihood、唯一 stationary carry fail-close、9×4 frozen LUT、leakage-only fallback、640k Markov cycles、32-row Source Data 和双任务记录。 | aggregate direct-minus-protocol cost `0.025622 [0.024873,0.026371]`；3 个非 control 场景 resolved 且有 nonleak override；首轮 scenario-seed pseudoreplication 已修复；11 gates、98 focused+adjacent、全量 `874 passed/4 skipped/4 个 R-N012 已知 failures`；R-N049 Mitigated/R-N050 登记，当前推荐更新为 T3.1.5。 |
| 2026-07-15 | T3.1.5 | Todo -> In Progress | 开始冻结 single-mode top-K lattice-coset truncated MAP 的 alias 排序、K 扫描、full periodic MAP 对照和确定性成本模型。 | 必须按每个 logical coset 累计 top-K likelihood，报告 LLR/LER 收敛与存储/乘加，不得借用 surface-code K-MWM 名称或只做 K=1 demo。 |
| 2026-07-15 | T3.1.5 | In Progress -> Done | 完成四陪集 joint top-K prefix sum、6 场景 4-seed 288k、K=1--128 full periodic MAP 对照、192-row Source Data 和 probability-domain cost proxy。 | 6/6 K=1 soft 非精确；联合门收敛 K=2--4；K=128 全场景浮点一致；8 gates、133 focused+adjacent、全量 895 passed/4 skipped/4 个 R-N012 已知 failures；R-N051 登记。 |
| 2026-07-15 | T3.2.1 | Todo -> In Progress | 开始审计现有 multi-round history/state 接口、observation budget、Bayesian sufficient state、causal update 和与 proposed controller 的公平比较合同。 | 必须消费相同 history/observation budget，禁止 truth leakage、evaluation tuning 或只实现 latest-outcome demo；不提前启动 T3.2.2。 |
| 2026-07-15 | T3.2.1 | In Progress -> Done | 完成 2L×2L joint periodic Bayesian filter、20-cycle observed-only history、same-prior final-static comparator、4 场景 4,096 episodes/81,920 cycles、32-row Source Data、Student-t CI、proper scores 与 128/256 grid convergence。 | aggregate static-minus-memory `0.303467 [0.291727,0.315207]`；9/9 gates、18 focused、167 adjacent、全量 `916 passed/4 skipped/4 个 R-N012 已知 failures`；修复 registry static/reference role 强绑并刷新 T3.1.1--T3.1.5 artifacts；R-N052 登记。 |
| 2026-07-15 | T3.2.2 | Todo -> In Progress | 开始实现 EWMA / Kalman adaptive MAP，并先审计 T1.3.4 现有 window/EKF 是否满足连续漂移强 baseline、公平 observation/update bandwidth 和多 seed 证据合同。 | 必须避免把旧单 trace alignment 直接包装为完成；与 T3.2.1/proposed estimator 对齐 causal observation 和成本口径。 |
| 2026-07-15 | T3.2.2 | In Progress -> Done | 完成 circular-moment full-covariance latest-window/EWMA/10-state Kalman、training-only 扩展网格、4 类 continuous drift、8-seed 157 万 paired samples、32-row Source Data 和 source-bound 证据。 | static-minus-EWMA/Kalman aggregate gain `0.009900 [0.009622,0.010179]` / `0.009933 [0.009658,0.010208]`；15 gates、37 focused、192 adjacent、全量 `953 passed/4 skipped/4 个 R-N012 已知 failures`；六份 registry-bound artifacts 重生成；R-N053 登记。 |
| 2026-07-15 | T3.2.3 | Todo -> In Progress | 开始冻结 sliding-window syndrome estimator 的窗口族、stride/update bandwidth、统计量、因果延迟、调参隔离和与 T3.2.2 latest-window 的语义差异。 | 不得把 T3.2.2 单一固定窗口行重复包装；必须在相同 observation budget 下做 multi-window/stride evidence 与成本核算。 |
| 2026-07-15 | T3.2.3 | In Progress -> Done | 完成 96-sample feature chunks 的增量 add/remove、384--1536 六窗、training-only score、4 类 continuous drift、8-seed 157 万 paired samples、32-row Source Data 和负结果边界。 | training/evaluation aggregate 均选 384；static-minus-selected `0.009740 [0.009529,0.009951]`，latest-minus-selected `0 [0,0]`；14 gates、25 focused、217 adjacent、全量 `978 passed/4 skipped/4 个 R-N012 已知 failures`；七份 registry artifacts 重生成；R-N054 登记。 |
| 2026-07-15 | T3.2.4 | Todo -> In Progress | 开始冻结 post-selection 的 score、threshold/coverage 网格、survival/rejection/conditional-error/total-cost 指标、truth 隔离与 diagnostic-upper-bound 语义。 | 必须把拒绝样本的成本和 coverage 全部报告；不得把 truth-based post-selection 当在线 decoder 或计入主增益。 |
| 2026-07-15 | T3.2.4 | In Progress -> Done | 完成 posterior-risk observed score、294,912 training-only threshold calibration、99.5%--50% 八档 survival、random/truth upper、157 万 evaluation samples、256-row Source Data 和四档 penalty cost。 | 90% aggregate raw/conditional `0.013785/0.001242`、capture `92.44%`，但 penalty=1 cost `0.101997`；14 gates、28 focused、117 adjacent、全量 `1006 passed/4 skipped/4 个 R-N012 已知 failures`；R-N055 登记。 |
| 2026-07-15 | T3.2.5 | Todo -> In Progress | 开始审计 observed g/e/leakage、quadrature phase、run-length saturating counters、normal/recovery/hold/fallback 状态和 parameter-bank action contract。 | 必须形成可执行确定性 FSM、显式阈值/优先级/失败分支和同 trace baseline；不得用 truth leakage 或只写状态图。 |
| 2026-07-15 | T3.2.5 | In Progress -> Done | 完成 3-bit 五态 FSM、真实 ParamBank、持续 local-safe conflict 修复、24-grid 117.96 万 training replay、384k evaluation、32-row Source Data 和负排序审计。 | static-minus-FSM `0.401710 [0.399014,0.404407]`，memoryless-minus-FSM `-0.179911 [-0.180782,-0.179041]`；15 gates、40 focused、131 adjacent、全量 `1046 passed/4 skipped/4 个 R-N012 已知 failures`；R-N056 登记。 |
| 2026-07-15 | T3.2.6 | Todo -> In Progress | 开始冻结 HMM/change-point 的 hidden regime、observed feature、causal filter、posterior calibration、latency/parameter budget和与后续 CNN 的公平比较合同。 | 必须输出 normal/burst/leakage/calibration-shift posterior，训练/评测隔离且不读取 hidden truth；不提前启动 T3.2.7。 |
| 2026-07-15 | T3.2.6 | In Progress -> Done | 完成 32-cycle observed summaries、四状态 full-covariance Gaussian HMM、same-emission memoryless ablation、3/3/8 disjoint seeds、54×10 training-only 网格、4,096 evaluation windows、Source Data 与术语源码绑定。 | HMM/memoryless accuracy `0.846191/0.660889`，NLL gain `0.401514 [0.366352,0.436676]`；15 gates、37 focused、51 focused+governance、131 adjacent、全量 `1083 passed/4 skipped/4 个 R-N012 已知 failures`；R-N057 登记。 |
| 2026-07-15 | T3.2.7 | Todo -> In Progress | 开始冻结 latest-outcome FNN/MF 的输入信息集、参数预算、动作空间、训练/选择/确认 split、same-seed pairing 和与 history model 的公平比较合同。 | 必须严格只消费当前 g/e/leakage outcome，禁止 hidden state、旧 history 或 evaluation tuning；不提前启动 T3.2.8。 |
| 2026-07-15 | T3.2.7 | In Progress -> Done | 完成 static390 latest-token front、精确 72,853 参数/72,266 dense MAC、5×300 epoch strict split、同 trace standard/旧 MF/exact MF/frozen NMF cutoff12/16、18,023-row Source Data 与 control-policy registry binding。 | cutoff12 `NMF-exact=-0.147464 [-0.386866,0.147532]`，cutoff16 `+0.540082 [0.231972,0.785521]`；13 gates、27 focused、72 governance、172 adjacent DLEnv；全量 `1083 passed/6 skipped/4 个 R-N012 已知 failures`；R-N044 更新/R-N058 登记。 |
| 2026-07-15 | T3.2.8 | Todo -> In Progress | 开始冻结 autonomous sBs 与 measurement-feedback 的协议原生 cycle、共同 wall-clock horizon、measurement/reset/active-control 事件和成本分解。 | 必须同时报告 per-cycle 与 per-microsecond，禁止用 0.7 timing 比例直接缩放 lifetime 或把 literature timing 写成目标板实测；不提前启动 T3.2.9。 |
| 2026-07-15 | T3.2.8 | In Progress -> Done | 完成 protocol-native nonselective autonomous/measurement 两路径、共同 700 us、cutoff12/16×三噪声、1,020 cycles、4,362-row Source Data、raw event ledger 和单位排序反转审计。 | per-cycle ratio `1.151287--1.346101`，per-us ratio `0.805901--0.942271`；17 gates、35 focused、161 adjacent；全量 `1083 passed/8 skipped/4 个 R-N012 已知 failures`；R-N059 登记。 |
| 2026-07-15 | T3.2.9 | Todo -> In Progress | 开始冻结有限时域 trajectory lookup control oracle 的 history alphabet、逐 history 独立 action 优化、训练/evaluation 分离、指数资源增长和 ansatz 内上界 contract。 | 必须与 decoder oracle 分名；禁止读取未来 outcome、用 evaluation 选 action 或只做穷举表壳而不优化真实 sBs trajectory；不提前启动 T3.2.10。 |
| 2026-07-15 | T3.2.9 | In Progress -> Done | 完成 15-node causal prefix tree、16-branch exact objective、open-loop 嵌套 warm start、两 families×3 restarts×550 epochs、cutoff16 frozen transfer、3,418-row Source Data、checkpoint replay、branch-burden 与指数资源审计。 | cutoff12 lookup/open/standard `0.815799/0.769403/0.396787`，cutoff16 lookup/standard `0.638688/0.559221`；20 gates、29 focused、75 governance、237 adjacent；全量 `1084 passed/10 skipped/4 个 R-N012 已知 failures`；R-N060 登记。 |
| 2026-07-15 | T3.2.10 | Todo -> In Progress | 开始冻结 PRL-inspired g/e/leakage 指数饱和递推的状态、参数、因果更新、训练/评估分离、定点化接口和与 run-length FSM/lookup 的比较合同。 | 必须使用可解释 `pi_{t+1}=a_m pi_t+(1-a_m)pi_m^inf` 或严格等价形式，禁止用隐藏大网络冒充递推、读取 truth/future 或只拟合单条 10g-10e-10g 展示轨迹；不提前启动 T3.2.11。 |
| 2026-07-15 | T3.2.10 | In Progress -> Done | 完成 15-control exponential recurrence、3×`300+250` exact optimization、cutoff12/16 standard/lookup/Q comparison、72-grid recurrence/24-grid FSM training-only selection、384k event evaluation、checkpoint replay 与 1,888-row Source Data。 | cutoff12 standard/recurrence/lookup `0.396787/0.784921/0.815799`；event recurrence/FSM/memoryless cost `0.073618/0.202829/0.022917`；19 gates、39 focused、85 governance、235 adjacent；全量 `1110 passed/11 skipped/4 个 R-N012 已知 failures`；R-N061 登记。 |
| 2026-07-15 | T3.2.11 | Todo -> In Progress | 开始冻结 memory-specific ablation 的 parent checkpoint、shuffle/truncation/reset/last-outcome-only 操作、同预算 retrain/frozen-view 分栏、多 cutoff/seed 配对和机制判定规则。 | 必须区分破坏 history 的 frozen intervention 与重新训练的 capacity comparator；禁止用单 agent、单 cutoff 或 evaluation 选消融强度来证明 memory；不提前启动 T4.1.1。 |
| 2026-07-15 | T3.2.11 | In Progress -> Done | 完成 5 frozen NMF×双 cutoff 的 prefix-consistent shuffle、L2/L4/L8 truncation、R2/R4/R8 reset、frozen latest-only，以及 5 个同预算重训 latest-only comparator；保存 28,230-row curves/action audit。 | cutoff12 full/retrained/frozen-latest `6.740785/6.888249/6.031675`，cutoff16 `7.708351/7.168269/8.271987`；15 gates、27 focused、131 governance/adjacent；全量 `1122 passed/12 skipped/4 个 R-N012 已知 failures`；两 cutoff 均不支持预注册 cross-cutoff mechanism，R-N062 登记。 |
| 2026-07-15 | T4.1.1 | Todo -> In Progress | 开始冻结 slow-loop model-selection 的共同 observation/history、参数/MAC/memory/latency envelope、training/validation/evaluation split、模型 family adapter 和 no-CNN-prior 选择规则。 | 必须在匹配预算与相同任务/metric 下比较 CNN/TCN、small GRU、Kalman/HMM、指数递推和 run-length FSM；禁止拼接不同 T3 metric 直接选 winner 或提前启动 T4.1.2。 |
| 2026-07-15 | T4.1.1 | In Progress -> Done | 完成 14-summary×8-window 同任务六族比较、TCN/GRU 各 5 restart、经典 training/validation grid、evaluation-blind selection、rolling-HMM exact cache、24,240-row Source Data 与 checkpoint/hash。 | validation HMM/TCN/GRU NLL `0.454975/0.476180/0.503134`，evaluation HMM `0.455711`；runner-minus-HMM CI `[0.046709,0.065742]`；13 gates、33 focused、79 governance、182 adjacent；全量 `1153 passed/14 skipped/4 个 R-N012 已知 failures`；R-N063 登记。 |
| 2026-07-15 | T4.1.2 | Todo -> In Progress | 开始冻结实验式 history 的逐 cycle observed schema、window alignment、recent action/LLR/run/deadline/update provenance、padding/mask/stale 语义和 truth-leakage denylist。 | 必须覆盖任务列出的全部输入族并逐字段追到真实 producer；hidden regime/state/logical truth/evaluation label 禁止进入 deployable payload，不提前启动 T4.1.3。 |
| 2026-07-15 | T4.1.2 | In Progress -> Done | 完成 10 groups/53 features/256-cycle observed-only history、真实 producer adapters、递归 denylist、padding/saturation/causal alignment 和 8×2,048-cycle stress replay。 | 17/17 gates、39 core、50 focused、138 governance、156 adjacent；全量 `1203 passed/14 skipped/4 个 R-N012 已知 failures`；16,384-row Source Data 覆盖 6 update statuses、5 FSM modes、g/e/leakage、deadline/pause/CRC/failure/stale/conflict/commit/FIFO/saturation；R-N064 登记。 |
| 2026-07-15 | T4.1.3 | Todo -> In Progress | 开始冻结 hybrid state output 的连续参数、regime posterior、leakage risk、recovery depth、uncertainty 与 parameter-bank recommendation schema。 | 输出必须有单位/范围/normalization/uncertainty/provenance 和 atomic-bank 语义；禁止直接输出逐周期纠错、hidden truth copy 或提前启动 T4.1.4。 |
| 2026-07-15 | T4.1.3 | In Progress -> Done | 完成 observed-only future hybrid state、T4.1.1 HMM bridge、periodic/Beta-run/block-bootstrap estimator、ParamMapper recommendation 和 ParamBank atomic stage/commit。 | 17/17 gates、16 core、27 focused、181 adjacent（2 skipped）；全量 `1230 passed/14 skipped/4 个 R-N012 已知 failures`；456 outputs、58 stage/commit、398 hold、五 profiles 全覆盖；horizon-risk 即时门误用已修复；R-N065 登记。 |
| 2026-07-15 | T4.1.4 | Todo -> In Progress | 开始冻结多目标 loss、任务间权重、training-only selection、state/oracle-gap/regime/uncertainty/fallback/update-cost calibration 与逐项 ablation。 | 必须分离训练/validation/evaluation、报告 proper scores 与 calibration，不得用 evaluation 调权或把 proxy risk 当物理校准；不提前启动 T4.2.1。 |
| 2026-07-15 | T4.1.4 | In Progress -> Done | 完成六项 typed loss、training-only scales、validation-only temperature/uniform-mix/uncertainty/fallback calibration、evaluation-blind scorer、448-row future alignment 与逐项 frozen-output ablation。 | 19/19 gates、25 focused、179 adjacent（2 skipped）；全量 `1255 passed/14 skipped/4 个 R-N012 已知 failures`；regime NLL `8.563501 -> 1.262520`、95% coverage `0.479167 -> 0.947421`；初版全 unsafe/零 update-cost 已修复；fallback recall/false rate `1/1` 负结果保留，R-N066 登记。 |
| 2026-07-15 | T4.1.5 | Todo -> In Progress | 开始冻结 offline teacher、distillation dataset、online student/safe baseline 的信息流、资源、checkpoint provenance、failure gate 与 teacher-student gap。 | teacher 可以使用 simulator truth/昂贵搜索但不得进入 online payload；online path 只能消费 observed history/state 并满足 deterministic budget；不得用 teacher action 直接冒充 student 或提前启动 T4.2.1。 |
| 2026-07-15 | T4.1.5 | In Progress -> Done | 完成 5-agent frozen NMF teacher hash 恢复、3×256×20 strict-split Source Data、3-restart/75-parameter validation-only 蒸馏、105-scalar hash-bound student 和 observed-health-only online API。 | 21/21 gates、28 focused、207 adjacent（2 skipped）；全量 `1283 passed/14 skipped/4 个 R-N012 已知 failures`；evaluation student/latest-only/zero-safe MSE `1.453624e-6/1.404389e-4/5.265504e-3`；初版 600 epoch 上限退化已修为 1200+plateau gate；全 health/leakage 零 residual fallback 通过，R-N067 登记。 |
| 2026-07-15 | T4.2.1 | Todo -> In Progress | 开始冻结 syndrome/quadrature phase/active parameter bank 到 periodic MAP LLR/action 的 parametric LUT、地址/插值/定点边界和确定 worst-case latency。 | 必须复用 canonical decoder convention 与真实 ParamBank；禁止 hidden truth、浮点运行时除法/指数、未定义越界地址或把软件 proxy 冒充 RTL/board timing；不提前启动 T4.2.2。 |
| 2026-07-15 | T4.2.1 | In Progress -> Done | 完成 active K/b + measurement covariance 反解、X/Z 257-entry Q9.12 ROM、10-bit ADC/8-bit address integer interpolation、8-bank full image artifact 和 5-stage pipeline。 | 20/20 gates、26 focused、210 adjacent；全量 `1309 passed/14 skipped/4 个 R-N012 已知 failures`；16,384 rows hard action mismatch `0`、mean/max LLR code error `0.387756/20`、interpolation/nearest ratio `0.004340`；初版 half-bin 地址偏差已修复，5-cycle/II=1/in-flight version latch 通过，R-N068 登记。 |
| 2026-07-15 | T4.2.2 | Todo -> In Progress | 开始冻结 g/e/leakage event 输入、饱和 run counters、normal/recovery/hold/reset-request/fallback priority、MAP action 到 Pauli/phase-frame 的原子更新和失败分支。 | 必须连接 T4.2.1 action/version 与 observed-only syndrome，覆盖同时事件、counter saturation、leakage/reset、invalid/CRC/stale/deadline；不得读取 truth 或提前启动 T4.2.3。 |
| 2026-07-15 | T4.2.2 | In Progress -> Done | 完成独立六态 observed-event FSM、六个 3-bit saturation counters、Pauli/phase-frame、1-cycle action register、8×128-cycle Source Data 和双任务记录。 | 20/20 gates、26 focused、98 focused+adjacent；全量 `1335 passed/14 skipped/4 个 R-N012 已知 failures`；修复 bank 0 全-I 时伪造 flip 的场景假设和 event/action 周期混名；6-cycle/II=1 仅为 software contract，R-N069 登记，当前推荐更新为 T4.2.3。 |
| 2026-07-15 | T4.2.3 | Todo -> In Progress | 开始冻结 OOD/leakage/stale/CRC/version/deadline fault taxonomy、conservative action、health flags、reason trace 与恢复条件。 | 必须复用 T4.2.1/T4.2.2 且把 event-local pilot 升级为完整可追溯 fallback；不得把 fail-closed software policy 冒充物理 recovery 或提前启动 T4.2.4。 |
| 2026-07-15 | T4.2.3 | In Progress -> Done | 完成 trusted image registry、14-bit health/integrity taxonomy、frame-hold/reset controller、16×256-cycle Source Data 和双任务记录。 | 20/20 gates、28 focused、126 focused+adjacent；全量 `1363 passed/14 skipped/4 个 R-N012 已知 failures`；修复缺失/损坏 MAP 时异常退出和 reported hash 自证可信风险；4,096 rows 覆盖全部 flags、组合 mask、8-bank/rollback、恢复/饱和、6-cycle/II=1；R-N070 登记，当前推荐更新为 T4.2.4。 |
| 2026-07-15 | T4.2.4 | Todo -> In Progress | 开始冻结 ADC/MAP/event/health/frame 全 fast path 的位宽、signedness、舍入/饱和、bit-accurate Python reference、float/高精度对照和 paired LER impact。 | 必须覆盖正常/边界/fault/version path，禁止只复用 T4.2.1 LLR ROM 或把 component quantization 冒充 end-to-end/RTL；不得提前启动 T4.3.1。 |
| 2026-07-15 | T4.2.4 | In Progress -> Done | 完成 integer-only MAP→health→event→frame reference、九类 arithmetic rules、四档 precision、87,040-code exhaustive Source Data、128-row paired LER Source Data 和双任务记录。 | 21/21 gates、15 focused、152 focused+adjacent；全量 `1378 passed/14 skipped/4 个 R-N012 已知 failures`；selected mean/max LLR error `9.46671e-5/0.00488281`，ΔLER `3.05176e-5 [-4.57764e-5,1.22070e-4]`；修复 health proxy 少算 16 bits 和 determinism 自比较问题；R-N071 登记，当前推荐更新为 T4.3.1。 |
| 2026-07-15 | T4.3.1 | Todo -> In Progress | 开始冻结 fast/event/slow 三种更新 cadence、clock/domain crossing、window/commit/recalibration 边界和 adaptation-lag 定义。 | 必须用执行 trace 量化 lag，连接 T4.2 fast path 与 host window/update；不得只画架构箭头或提前启动 T4.3.2。 |
| 2026-07-15 | T4.3.1 | In Progress -> Done | 完成 exact cadence runtime、真实 scheduler/ParamBank/T4.2 trace、双口径 8000-phase Source Data、minute/end-run due、age/cadence 集成修复和双任务记录。 | 14/14 gates、26 focused、106 adjacent；全量 `1404 passed/14 skipped/4 个 R-N012 failures`；first-influenced/full-post-change `1.000--20.995/11.235--31.230 ms`；修复 64-cycle stale pilot 与 4000-cycle slow cadence 冲突，R-N072 登记，当前推荐更新为 T4.3.2。 |
| 2026-07-15 | T4.3.2 | Todo -> In Progress | 开始实现带 version/CRC/timestamp/CAS 的完整 inactive-bank image、cycle-boundary atomic switch、ack/readback 与 hysteresis。 | 必须复用 T4.3.1 cadence/T4.2 trusted image，证明 partial/torn/stale/race payload 永不激活；不得提前启动 T4.3.3。 |
| 2026-07-15 | T4.3.2 | In Progress -> Done | 完成 thread-safe full-image atomic bank、三层 CRC/SHA、cycle timestamp、CAS/anti-replay、两窗 hysteresis、safe-boundary/min-residency commit、ack/readback、public API 与双任务记录。 | 17/17 gates、7518 rows、37 focused、115 adjacent；全量 `1441 passed/14 skipped/4 个 R-N012 failures`；穷举 3745 prefix 和 3745 byte flip，修复 header integrity 与公开属性锁遗漏；R-N073 登记，当前推荐更新为 T4.3.3。 |
| 2026-07-15 | T4.3.3 | Todo -> In Progress | 开始构建闭环 fault-recovery harness，连接 T4.3.1 cadence、T4.3.2 atomic bank 与 T4.2 fast fallback，注入 drift/burst/leakage、host timeout、通信中断、pause/jitter 和更新竞态。 | 必须证明无未定义 action、active image/trace/reason 可追溯并审计 recovery/rollback 边界；不得把 software stress 冒充 RTL/board 或提前启动 T4.4.1。 |
| 2026-07-15 | T4.3.3 | In Progress -> Done | 完成 closed-loop supervisor、ack uncertainty、host timeout、post-commit guard、monotonic LKG republish、cadence refresh、8×4×23996-cycle campaign、17-gate artifact 和双任务记录。 | 767872 cycles 的 undefined action/blocking correction/frame overflow 均0；19 focused、124 adjacent；最终全量 `1460 passed/14 skipped/4 个 R-N012 failures`；修复 freshness、stale recovery flag、unregistered same-version image 与 artifact hash 耦合；R-N074 登记。 |
| 2026-07-15 | T4.4.1 | Todo -> In Progress | 开始冻结 bounded residual RNN/GRU teacher 的 15-output scope、nominal action、hard bounds、data split、from-scratch convergence、checkpoint provenance 与 failure branch。 | 必须复用 T2.3.7/T3.2.9/T4.1.5 teacher evidence，不能把旧 checkpoint 重命名为新 teacher；不得提前启动 T4.4.2。 |
| 2026-07-15 | T4.4.1 | In Progress -> Done | 完成 3 个 fresh GRU restart 的真实 Feedback-GRAPE 训练、15-output hard-bound action、strict split、validation-only selection、parent hash non-reuse、双 cutoff held-out、1,074-row Source Data 与 checkpoint reload。 | 21/21 gates；validation gain `0.284132/0.281901/0.284383`，primary/confirmation score gain `0.253603/0.141557`；focused 14、邻接 102；全量 `1472 passed/17 skipped/4 个 R-N012 已知 failures`；601/811 在 epoch 320 达峰并明确不声明全局收敛，R-N075 登记。 |
| 2026-07-15 | T4.4.2 | Todo -> In Progress | 开始对 selected fresh teacher 的固定 g/e/run/alternating/leakage-proxy 序列提取 hidden state、15-control response、belief proxy、有效记忆长度、指数饱和与 `p(g)` 关系。 | 必须只做 post hoc analysis，不用 evaluation 结果回调 teacher；leakage 若仍无训练语义必须标 proxy/OOD，不得提前启动 T4.4.3。 |
| 2026-07-15 | T4.4.2 | In Progress -> Done | 完成 frozen teacher 的 10×128 native trace、20-half-cycle p(g)、24/8 trajectory-disjoint belief probe、PCA、30 参数指数 fit、双向 impulse/Jacobian 和 leakage reset+nominal OOD proxy。 | 17/17 gates、2,089 rows、focused 13、邻接 87、全量 `1484 passed/19 skipped/4 个 R-N012 已知 failures`；hidden/control 95% PC 均1，hidden probe `R²=0.667797`，28/30 高拟合并保留 virtual-rotation 反证，R-N076 登记。 |
| 2026-07-15 | T4.4.3 | Todo -> In Progress | 开始依据 T4.4.2 的 1--2 PC、10--12 half-cycle memory 和 virtual-rotation 反证，冻结可解释指数递推 student 的状态维数、15-output head、strict split、拟合/失败 gate 与 artifact。 | 不直接复用 T4.1.5 旧 student 充数；不得用 evaluation 选维度/超参数，leakage 仍需 fail closed，不得提前启动 T4.4.4。 |
| 2026-07-16 | T4.4.3 | In Progress -> Done | 完成 1/2/4-state×3-restart、各 900 epochs 的 strict-split teacher-action distillation、validation-only 选维、strong comparators、hash-bound NumPy online artifact、58,356-row Source Data 与双任务记录。 | 16/16 gates；选中 4 states/95 scalars，validation/evaluation MSE `5.648504e-6/6.083136e-6`；32 focused、83 adjacent、34 governance；全量 `1515 passed/21 skipped/4 个 R-N012 failures`；6 个 cap hits 明确保留，R-N077 登记。 |
| 2026-07-16 | T4.4.4 | Todo -> In Progress | 开始冻结 teacher-student physical gain-retention protocol，统一 standard、MF、teacher、handcrafted recurrence、selected student 和 control oracle 的初态、噪声、trajectory seeds、lifetime/fidelity/`p(g)`/e-leakage burden 与成本口径。 | 不得以 T4.4.3 imitation MSE 代替 physical gain；必须预注册 retention threshold 与 strong/falsified failure branch，不得提前启动 T4.4.5。 |
| 2026-07-16 | T4.4.4 | In Progress -> Done | 完成全新 paired seeds 的 cutoff12/16 10-cycle physical replay、全部五个 MF agents、teacher/handcrafted/student、独立 exact 2-cycle control-oracle、90% point/CI gate、显式 burden/cost 与双任务记录。 | 18/18 gates、448 rows、17 focused、129 focused+adjacent、35 governance；全量 `1531 passed, 23 skipped, 4 failed`，失败仅为 R-N012；最低 point/CI lower `0.981457/0.944501`；保留 cutoff-dependent MF reversal 和 leakage-null，R-N078 登记，当前推荐更新为 T4.4.5。 |
| 2026-07-16 | T4.4.5 | Todo -> In Progress | 开始只读消费 T4.4.1--T4.4.4 machine gates，冻结 strong/falsified branch、允许/禁止 claim、MAP-LUT fallback 和后续 evidence gates。 | 通过 retention 不得覆盖 MF 排名反转、leakage/OOD/long-horizon/硬件缺口；分支判定必须机器可验证，不得重新跑 evaluation 选结论。 |
| 2026-07-16 | T4.4.5 | In Progress -> Done | 完成 hash-bound evaluation-free branch freeze、72 parent gates、7 file bindings、8 evidence predicates、112-row claim/revocation ledger 和九类逐一 failure mutation。 | 11/11 gates、17 direct、129 adjacent/protocol tests；全量在 605.7 s 上限运行至 19%，终止前未见失败但不记为通过；激活 qualified student-retention 并保留 MF reversal/七类 claim 禁区，R-N079 登记，当前推荐更新为 T5.0.1。 |
| 2026-07-16 | T5.0.1 | Todo -> In Progress | 开始建立逐来源、逐趋势的 reproduction registry，区分 main/cross-validation/secondary、数值/方向目标、tolerance、calibration/holdout 和禁止 transfer。 | 只复用已锚定的一手/secondary 来源与现有 reproduction artifacts；不得为填表伪造实验数值或把 secondary 升入 sBs 主排名。 |
| 2026-07-16 | T5.0.1 | In Progress -> Done | 完成 7-source、8-local-anchor、6-artifact-binding、14-target literature trend registry、human table、52-row Source Data、protocol hierarchy 与双任务记录。 | 17/17 gates、15 direct、52 focused+governance、224 adjacent tests；5 pending、2 reference、1 reporting-only 均未计作复现，high/low noise-transfer 与 NMF cutoff reversal 保留；R-N080 登记，当前推荐更新为 T5.0.2。 |
| 2026-07-16 | T5.0.2 | Todo -> In Progress | 开始按 T5.0.1 冻结的 calibration/holdout contract 执行跨保真度独立验证，优先选择一个主线趋势和一个 secondary 解析趋势，测试参数不得回流调参。 | 必须使用未参与 T2.3.3/T5.0.1 阈值建立的参数范围或协议点；pending 只有对应独立 gate 通过才能升级，secondary 仍不进入 sBs 主排名。 |
| 2026-07-16 | T5.0.2 | In Progress -> Done | 完成 disjoint `2.5/10.25/11.75 dB` 四-fidelity holdout、4 fresh seeds/点、252-point P-Steane 双路径解析回归、291-row Source Data、human report、protocol contract 与双任务记录。 | task 6/6 gates；main 明确 FAIL（`10.25 dB` z=`2.293338>2`）、secondary PASS；118 focused、319 adjacent tests；未重选失败点或改写 T5.0.1 snapshot，R-N081 登记，当前推荐更新为 T5.1.1。 |
| 2026-07-16 | T5.1.1 | Todo -> In Progress | 开始盘点并冻结完整 comparison set 的 canonical role、实现绑定、输入可见性、deployability、预算和 ranking eligibility。 | 必须覆盖任务板列出的全部 comparator，区分 decoder/control/channel oracle 与主/secondary；缺失实现需 fail closed，不得用同名弱 baseline 或 Knill/P-Steane 填入 sBs 主排名。 |

## 任务改进说明(含参考文献)

最有价值的是 2411.05262 的低成本噪声传递模型和 2401.02022 的近最优逻辑通道界；其余四篇主要用于协议 baseline、计算量折中和论文实验叙事。

| 论文 | 类型与核心结果 | 对本项目的价值 | 优先级 |
| --- | --- | --- | --- |
| [2605.08009：Error Correction of Beamsplitter-Generated Entangled GKP States](https://arxiv.org/abs/2605.08009) | 真实 trapped-ion 双模实验；四种 Bell 态平均保真度约 69%，QEC 将纠缠态寿命平均延长约 `2.0(2)` 倍；单轮 QEC 约 `500 μs` | 不作为 single-mode baseline，但非常适合借鉴论文叙事：分别报告 Pauli lifetime、QEC on/off、wall-clock、轮次成本、reset recoil 和并行控制代价 | 中 |
| [2604.08247：Optimized GKP Error Correction via Tunable Preprocessing](https://arxiv.org/abs/2604.08247) | 理论与数值工作；用可调压缩参数 `(a,b)` 构造 P-Steane，依据 data/ancilla 噪声比例主动整形输出噪声；`2a=b` 给出重要最优条件 | 可作为 Steane-secondary 路线的 gain-scheduled protocol baseline。主机估计噪声比例，输出 `(a,b)` 参数库；但 FPGA 只能选择参数，不能实现物理 squeezing | 中高 |
| [2411.05262：Noise Transfer Approach to GKP Quantum Circuits](https://www.mdpi.com/1099-4300/26/10/874) | Heisenberg-picture 解析方法，把 GKP 信号、连续涨落噪声和离散逻辑错误分开传播；可处理 loss、measurement efficiency、feedforward gain | 非常适合成为 syndrome model 与 Fock model 之间的低成本中间保真度模型，并用于逐组件 noise budget 归因 | 最高 |
| [2505.14775：Performance analysis of GKP error correction](https://arxiv.org/abs/2505.14775) | 解析比较 Knill 与 Steane；证明 Steane 是特定资源态下的 Knill 特例；qunaught + beamsplitter 的 Knill 方案保留更好的对称 squeezing；数值等价误差低于 `10^-8` | 增加 Knill/qunaught、Steane/ME-Steane 的协议对照和解析回归测试，但因项目以 sBs 为主，应放 Supplement 或 secondary reproduction lane | 中 |
| [2401.02022：The Near-optimal Performance of Quantum Error Correction Codes](https://arxiv.org/abs/2401.02022) | 基于 QEC matrix 和 transpose/Petz channel 构造无需优化的 near-optimal channel fidelity，并给出 optimal recovery 的双边界；可扩展到高能 GKP | 应作为新的“channel-recovery bound”，衡量实际 sBs、teacher/student 与编码本身潜在性能之间还有多大差距 | 最高 |
| [2510.06531：Approximate maximum-likelihood decoding with K minimum weight matchings](https://arxiv.org/abs/2510.06531) | 用前 `K` 个 MWM 近似 MLD，通过 `K` 调节精度和计算量；主要实验对象是 surface/surface-GKP | 原算法超出 single-mode 范围，不应直接实现；但可借鉴为 top-K lattice-coset MAP：每个逻辑陪集只累加最可能的 K 个 lattice aliases，并测量 K—LER—资源 Pareto | 中高 |

建议对任务板进行一次“小幅 v2.2 补强”，不需要再次整体重构：

1. 在 `T1.4.5` 增加第三类上界术语：

   - decoder oracle；
   - control oracle；
   - channel-recovery bound，即 QEC-matrix/Petz 近最优恢复界。

   后者不是可部署 decoder，不能与 oracle MAP 混称。

2. 在 M2.3 增加 noise-transfer surrogate：

   - 分离 lattice signal、continuous fluctuation 和 discrete logical jump；
   - 传播 covariance、loss、measurement efficiency 和 feedforward gain；
   - 与 syndrome model、Fock model 交叉验证；
   - 明确适用门槛：论文指出约 `10 dB` 以上 squeezing 时近似更可靠，低 squeezing 下会出现 clipping 和 state-dependent 偏差。

3. 在 M3.1 增加 top-K lattice-coset MAP baseline：

   - `K=1`：近似 nearest/minimum-distance 路线；
   - 有限 `K`：硬件友好的 truncated MAP；
   - 足够大 `K`：逼近当前 periodic Gaussian MAP；
   - 报告 K、LER、LUT/BRAM/DSP、latency 和数值误差。

4. 在 T5.0.1 的 reproduction table 中加入：

   - Knill/Steane 数值等价；
   - qunaught-Knill 与标准 Bell-resource Knill 的 squeezing/displacement 趋势；
   - P-Steane 在不同 data/ancilla 噪声比下的参数趋势。

   这些只作为 secondary protocol evidence，不进入 sBs 主图竞争。

5. 在 M5.3 增加 near-optimal channel fidelity：

   - 小 cutoff 下与 SDP optimal recovery 对齐；
   - 中高 cutoff 使用 QEC-matrix/Petz bound；
   - 报告实际控制器到该 bound 的 gap；
   - 如果 bound 本身改善但实际 controller 没改善，应归因于 recovery/controller 不充分，而不是编码失效。

6. 利用 2605.08009 补强既有实验叙事，不新增多模任务：

   - 六个 Pauli eigenstates/各逻辑方向 lifetime；
   - QEC on/off；
   - cycle 数与真实 wall-clock 同时报；
   - reset、recoil、测量和并行模式带来的成本；
   - 明确不同平台的 `500 μs` 不能作为本项目 FPGA deadline。

最重要的范围控制是：不引入 surface-GKP 主线，不把 P-Steane 的物理 squeezing 写成 FPGA 能实现的功能，也不把 Petz bound 写成实际 decoder 性能。按这个边界吸收后，六篇论文能够增强项目的物理建模、baseline 强度和论文说服力，而不会把当前 v2.1 再次扩散成协议大全。

本轮仅完成论文筛选与任务映射分析，尚未修改 [task_board.md](D:/Codes/Quantum/CNN_FPGA_GKP/docs/task_board.md)。
