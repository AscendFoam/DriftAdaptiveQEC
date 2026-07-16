# T2.2.2 协议原生 ancilla/readout/reset error model

**日期：** 2026-07-14  
**状态：** Implemented effective layer  
**实现：** `physics/protocol_ancilla_errors.py`  
**机器证据：** `docs/t2_2_2_protocol_ancilla_validation.json`

## 1. 结论与边界

本任务没有把所有协议压成同一个二值 demo。实现由两条互不共享观测字母表的
状态机组成：

1. sBs 主协议继续使用 `K_gg/K_ge/K_eg/K_ee`、按 X 后 Z 的 constituent
   顺序和 observed `g/e/leakage`；T2.0.3 的 hidden/readout/reset kernel 被原样
   复用，再叠加 constituent × stage 的 bit/phase fault；
2. sharpen--trim 交叉验证协议单独使用四轮
   `q-peak -> p-peak -> q-trim -> p-trim`、原生 `+y/-y` 观测、
   outcome-dependent feedback 和 conditional `pi/2` reset。

Steane、Knill/qunaught 与 P-Steane 没有可执行 simulator。它们只在 registry 中
登记允许扫描的 noise-shaping 参数，并显式设置 `executable=false`。

该层是有一手机制锚的 **effective stochastic model**，不是 cavity--transmon
master equation，也没有把项目假设的 fault rate、confusion matrix 或 reset
success 包装成装置标定值。

## 2. 一手证据映射

### 2.1 sBs

- Sivak 2023 补充材料第 1121 行：phase flip 在 Z-basis readout、virtual
  rotation/idle 与 SBS circuit 中是 fault-tolerant 或小误差，不应被简单建模为
  Z-basis outcome toggle；
- 第 1123、1157 行：big conditional displacement 中段的 bit flip 可能形成
  logical error；readout misclassification 会驱动错误 virtual rotation，文献给出
  0--0.6 rad 的机制范围；
- T2.0.3 已冻结 full hidden `g/e/f/higher` -> observed
  `g/e/leakage` -> conditional reset kernel，本任务不另造第二套 sBs readout。

### 2.2 sharpen--trim

- Campagne-Ibarcq 2020 第 35、599、623--626、637--671 行冻结四轮顺序、
  `sigma_y` measurement、`|+y>/|-y>` 与 outcome-conditioned feedback/reset；
- 第 735 行说明 phase flip 与 readout error 在该 Markovian feedback 中都使
  feedback displacement 方向错误；
- 第 773 行给出 peak-sharpening interaction 中间四分之一到四分之三的 bit-flip
  logical-error window；
- 第 411 行说明 `|f>` leakage 会绕开 g/e control 并持续到随机 relaxation，
  因而必须保留 hidden leakage carry，而不能把它伪装成第三个可部署观测标签。

## 3. sBs fault overlay

### 3.1 Stage-resolved fault

每个 X/Z constituent 都有三个显式 stage：

| stage | bit flip | phase flip |
| --- | --- | --- |
| `small_cd` | 翻转该 constituent 的 g/e outcome，并加入有符号小回作用 | 不翻转 Z-basis outcome，只加入有符号小回作用 |
| `big_cd` | 翻转 outcome；按显式条件概率产生 hidden logical backaction | 不翻转 outcome，保留小回作用 |
| `readout` | 作为 readout 前的 g/e 翻转进入既有 full classifier | Z-basis outcome 不翻转，连续回作用为 0 |

多次 bit flip 按 parity 合成，因此同一 constituent 两次翻转会取消 outcome toggle，
但事件 provenance 仍完整保留。所有概率、backaction scale 和最大旋转角必须由调用方
显式传入并带 `parameter_provenance`。

### 3.2 信息流

`SBSFaultOverlayTruth` 只供 simulator 验证，包含 fault stage、原始/故障后 Kraus
label、hidden logical/continuous backaction、misclassification 和 virtual-rotation
error。可部署记录只转发 T2.0.3 的 observed syndrome、reset action 和 observed
run counters；不会暴露 fault event、hidden state 或 logical truth。

读出误判只在 hidden g/e 与 observed class 不同的情况下成立。误判的 virtual
rotation 在 `[-rotation_max,+rotation_max]` 中采样；范围是机制敏感度参数，
不是自动等同于实验分布。

### 3.3 Reset 与 leakage

overlay 直接调用 `SBSObservationResetModel`，因此 full 4×3 confusion、f/higher
聚合 leakage、reset success/failure、hidden carry 与 observed run-length 语义均不
被旁路。测试用 persistent `higher` hidden state 证明连续两轮仍为 observed
`leakage`，而 deployable view 无法看到 `higher` 真值。

## 4. sharpen--trim native state machine

### 4.1 四轮动作

四轮按固定顺序循环。peak round 的小 feedback 为

`a * (sign * peak_fraction + asymmetry_fraction)`，

trim round 为 `sign * a * trim_fraction`。轴由 q/p round type 决定。
每轮只读出 `+y/-y`，并据此选择相应 conditional `pi/2` reset action。

确定性 protocol frame 每四轮闭合：

| round | intrinsic logical frame | cumulative frame from zero |
| --- | --- | --- |
| q peak | Z | Z |
| p peak | X | XZ |
| q trim | X | Z |
| p trim | Z | I |

随机 bit-flip logical backaction 是不可观测 physical truth，**不会**被偷偷写进
deployable Pauli frame。否则控制器会得到它本来不知道的故障标签。

### 4.2 Bit/phase/readout

- 在 `+y/-y` basis 中，单个 bit 或 phase flip 都切换 outcome；同时发生时按
  parity 抵消；
- 只有 peak round 且 bit flip interaction fraction 位于 `[0.25,0.75]` 时记录
  logical backaction；p-peak 映射 X、q-peak 映射 Z；
- full 3×2 confusion matrix 映射 hidden `+y/-y/leakage` 到 observed
  `+y/-y`，因此 leakage 没有被增设成可部署第三类；
- observed outcome 决定 feedback sign 和 reset action；相对 ideal outcome 的方向
  错误只保留在 truth diagnostics。

### 4.3 Reset failure 与 leakage carry

correct-observation reset、wrong-sign reset 与 leakage reset 分别有独立 success
probability。失败时 hidden `+y/-y/leakage` 进入下一轮并覆盖该轮理想 outcome；
hidden leakage/reset-failure run 只存在 simulator memory，不进入 deployable record。

## 5. Secondary protocol fail-closed registry

| protocol | executable | 只允许扫描 |
| --- | --- | --- |
| Steane | false | data variance、ancilla variance |
| Knill/qunaught | false | resource squeezing、homodyne variance |
| P-Steane | false | `a`、`b`、data/ancilla noise ratio |

这些字段只表达理论/noise-shaping 参数空间，不代表实现了 state preparation、
measurement circuit、hardware timing 或主协议 ranking。

## 6. 验证与反简化审计

### 6.1 Direct/adjacent tests

`tests/test_protocol_ancilla_errors.py` 共 27 项，覆盖：

- deterministic small/big/readout-stage fault 与双翻转 parity；
- phase flip 不切换 sBs Z-basis outcome；
- readout misclassification/rotation bound；
- sBs leakage/reset persistence 与 deployable/truth schema；
- seed replay 与 prefix stability；
- sharpen--trim 四轮 action/frame、bit/phase 双故障抵消；
- 20,000-round middle-window logical-backaction rate；
- reset failure 与 leakage 跨轮 carry；
- 20,000-round leakage-row 3×2 confusion calibration；
- protocol alphabet negative mixing、secondary non-execution 和非法配置；
- 10,000-sample analytic rate validation 与 JSON round trip。

相邻 sBs suite 共 115 项通过，说明新 overlay 没有破坏 T2.0.3--T2.0.6。

### 6.2 Production seeded validation

正式产物使用 80,000 sBs cycles 与 80,000 sharpen--trim rounds：

| check | expected | observed |
| --- | ---: | ---: |
| sBs X big-CD bit rate | 0.020000 | 0.020175 |
| sBs hidden logical backaction rate | 0.010000 | 0.009875 |
| sBs phase-induced Z-basis toggle | 0 | 0 |
| sharpen feedback mismatch | 0.071648 | 0.073788 |
| sharpen bit middle-window rate | 0.010000 | 0.010325 |
| sharpen bit logical-backaction rate | 0.005000 | 0.005213 |

所有预注册统计项在 5 standard errors 内，secondary registry 全部保持
`executable=false`。

## 7. Claim boundary

允许声称：

- 已实现 sBs stage-resolved effective ancilla fault overlay；
- 已实现与 sBs 字母表隔离的 sharpen--trim 原生四轮 error-flow state machine；
- readout/reset/leakage carry、truth/deployable schema 和 secondary non-execution 已测试。

禁止声称：

- fault probability、3×2/4×3 confusion、reset success 已由目标装置标定；
- stochastic backaction 等同于 cavity--transmon master equation；
- 复现文献 65× sensitivity、实验 logical error budget 或真实 lifetime；
- Steane/Knill/qunaught/P-Steane 已实现或进入主排名；
- 已包含 T2.2.3 的 DAC/AWG、pulse miscalibration、latency 与 active displacement
  imperfection。
