# T2.0.1 主/次参考协议层级冻结

**日期：** 2026-07-14  
**状态：** Frozen contract；T2.0.2--T2.3.6、T2.3.8 与 T-RISK-20260714-01 已实现 error-space、observation/reset、reference cycle FSM、位移/occupancy/correlation、mixed-state stream、control memory、fast/rare Monte Carlo、finite-squeezing、ancilla fault、control-imperfection、generic finite-Fock、completed analytic SBS one-round、noise-transfer、axis-resolved cross-fidelity、quadrature contract、joint cavity--two-level-ancilla differentiable trajectory、Feedback-GRAPE gradient validation 与 current-host training resource envelope；ranking 和 device fidelity 仍 fail closed  
**机器可读镜像：** `docs/protocol_hierarchy.json`

## 1. 结论

后续主数字孪生只使用 **measurement-feedback sBs low-rank dissipation**。2020 年 sharpen–trim/measurement-feedback 协议是具有一手实验来源的异构交叉验证，不与 sBs 合并。Knill/qunaught 与 ME/P-Steane 当前只有仓库 secondary 证据，只允许进入小规模解析回归、secondary reproduction 或 Supplement。

这一层级首先冻结“必须实现什么”和“不能混用什么”。`PROTO-SBS-MAIN` 已实现 coarse-grained error-space transition、hidden/observed/reset、Table S3 文献参考 FSM、位移与 occupancy/correlation 趋势、完整 `DriftState` stream、observed-only multi-round memory、million-cycle vectorized/rare-stratified Monte Carlo、finite-squeezing 分解、constituent × stage ancilla fault、control-imperfection layer、finite-cutoff completed analytic SBS X→Z one-round、noise-transfer surrogate、Fock/effective/noise-transfer/direct-syndrome 的 chart-qualified axis-resolved cross-fidelity、使用独立 Puviani Table S1 timing 的 joint cavity--two-level-ancilla differentiable trajectory、Feedback-GRAPE gradient/resource/ranking，以及 T4.4.1 fresh bounded-residual GRU teacher、T4.4.2 frozen analysis、T4.4.3 low-dimensional student 和 T4.4.4 physical gain retention；另保留 generic single-oscillator finite-Fock density/channel 层。旧 Fourier-p 失配已归因为 decoder/canonical 混用并修复，legacy 路径继续保存为负证据。`PROTO-SHARPEN-TRIM-XVAL` 已实现独立四轮 `+y/-y` ancilla/readout/reset effective state machine。pulse Hamiltonian、multilevel transmon/leakage/SPAM、device calibration、long-horizon/OOD student retention、真实板卡 timing 和实验 raw-data 定量拟合仍未实现。两个 secondary 协议仍为 `contract_only_not_implemented`。

## 2. 协议对照表

| ID | 层级 | 周期口径 | 观测 | 动作 | 当前明确不可模拟/未注册项 |
| --- | --- | --- | --- | --- | --- |
| `PROTO-SBS-MAIN` | 唯一主数字孪生 | 两个 rank-2 的 X/Z constituent 组成一个 rank-4 full cycle；文献参考分别为 `4.924 us` 和 `9.848 us`，目标板实测为 `null` | constituent 为 `g/e`；full cycle 必须为有序对 `gg/ge/eg/ee`；`f` 仅属 reset/leakage control branch | 四层 sBs `U_empty`、测量/反馈 reset、`g/e/f` virtual rotation、X/Z 切换、Pauli-frame tracking | 脉冲/Hamiltonian、装置专用 RL 参数、真实 confusion matrix/校准角、实板时延、无限维精确动力学 |
| `PROTO-SHARPEN-TRIM-XVAL` | 一手交叉验证 | 两个 peak-sharpen + 两个 envelope-trim 的四轮 block；单轮和整块时长均为 `null` | 每轮测量 `sigma_y`，结果 `+y/-y` | conditional displacement、sharpen recenter、trim `a/2` shift、`pi/2` reset、逻辑 frame | 精确周期时长、完整脉冲链、`g/e/f` reset、与 rank-4 sBs 的直接等价 |
| `PROTO-KNILL-QUNAUGHT-SECONDARY` | secondary reproduction | protocol-specific，时长为 `null` | 当前未注册；不得臆造 | 一手全文入库后才允许小规模解析/数值回归 | optical resource、homodyne/beamsplitter、设备时序、FPGA 物理控制 |
| `PROTO-ME-PSTEANE-SECONDARY` | secondary reproduction | protocol-specific，时长为 `null` | model-level noise-ratio estimate | host 可选已验证 `(a,b)` 参数库；FPGA 至多选择索引 | 物理 squeezing、ancilla 制备、设备时序、脉冲生成 |

<!-- protocol-id: PROTO-SBS-MAIN -->
<!-- protocol-id: PROTO-SHARPEN-TRIM-XVAL -->
<!-- protocol-id: PROTO-KNILL-QUNAUGHT-SECONDARY -->
<!-- protocol-id: PROTO-ME-PSTEANE-SECONDARY -->

## 3. 主协议的非简化语义

### 3.1 通道与周期

Sivak 等的 full channel 是 `R_Delta = R_Delta_X o R_Delta_Z`：两个 rank-2 constituent 组合为 rank-4 通道。`4.924 us` 是一个 X 或 Z constituent；完整 X+Z cycle 是 `9.848 us`。后续代码、图表和 provenance 必须显式写 constituent/full，不能把前者缩写成完整 cycle，也不能把任一文献数字写成廉价 FPGA 实测。

### 3.2 观测不是单个 g/e

完整周期 Kraus 标签是 `K_gg/K_ge/K_eg/K_ee`。由 `K_ge=K_g^Z K_e^X` 可知，标签字符顺序是 `(Z,X)`，真实执行顺序则是右侧 X constituent 先、左侧 Z constituent 后；因此 `K_ge` 在按时间的 `(X,Z)` 表示中是 `(e,g)`。`gg` 表示主要保持同一 error subspace，`ge/eg` 表示逐级转移，`ee` 表示两级转移；这不是“投影到唯一 syndrome 后一步恢复”的稳定子简化。原文还明确说明完整解释必须看 outcome pair，而不是孤立的 `g/e`。

实验控制器能识别 `g/e/f` 并给出不同 reset/virtual-rotation calibration，但 `f` 不属于四个理想 full-cycle Kraus 标签。T2.0.3 必须把隐藏 error state、观测分类和 reset/leakage state 分开。

### 3.3 动作与 frame

主协议的 action contract 包含 sBs `U_empty`、测量反馈、条件 reset、`g/e/f` 虚拟旋转、X/Z 象限切换和 deterministic logical flip 的 Pauli-frame tracking。只实现一个 syndrome lookup 或一次性 nearest-grid shift 不满足本协议。

### 3.4 位移故障趋势

Fig. 4(c) 的关键不是“位移越大 syndrome 越长”，而是位移到最近逻辑操作的距离：`epsilon/l_S=0` 与 `0.5` 都接近合法逻辑操作，`0.25` 才是 large-distance midpoint。T2.0.5 用周期距离生成 recovery depth，经过 T2.0.2 transition 与 T2.0.3 observed/reset kernel 得到 position-fault `K_eg/K_eg/...` 和同象限 e-run。4096-shot 预注册 sweep 在 `0.25` 取得最大 observed run `4.883 [4.846,4.919]`，两侧 Spearman 为 `1/-1`，未受影响 X 象限 `P_e` 最大为 `0.0273`。该结果是 qualitative effective-model trend，不是 Fig. 4(c) 数字化或装置标定。

### 3.5 Occupancy 与 correlation

T2.0.6 把 hidden truth 与 observed-only estimator 分离。600×1200-cycle run 中 hidden occupancy 为 `0.813565 [0.811663,0.815456]`，仅由 `P([gg]^n)=a lambda^n` 得到 `0.813524 [0.811288,0.815992]`，`p_err=0.132011`；S4E 的 `p_err^2` 一阶系统界另列，不被统计 CI 吞并。对 lags 40--200，mean non-g correlation 从 `0.002976` 降到 `-0.000192`，paired difference CI `[0.001684,0.005058]`；post-selection 保留 84.5% trajectories。该步骤是离线去除含连续 leakage run 的 trajectory，不等于实现在线 leakage controller。

### 3.6 Mixed-state syndrome stream

T2.1.1 的 `physics/syndrome_stream.py` 逐周期消费完整 `DriftState`。它显式分开相关 Gaussian mixture displacement、`sqrt(eta)` residual attenuation、loss-environment noise、modular truth、measurement noise、coarse recovery/leakage 与 logical Pauli truth；输出 analog/residual q-p syndrome、按 `(X,Z)` 的 `g/e/leakage`、`(0,pi/2)` phase 和 observed runs。deployable record 不含 hidden regime、outlier component、leakage truth、recovery depth 或 logical label；完整 provenance 与 truth 在独立 schema 中。

该层不是把 T2.0.3 的 full configurable ancilla kernel 偷换成装置模型。默认 depth/recovery/leakage law 和 higher duration仍为 project assumptions，且 controller memory 与大规模 Monte Carlo 分别留给 T2.1.2/T2.1.3。详细公式、字段和 21 项 direct tests 见 `docs/syndrome_stream_model.md`。

### 3.7 Observed-only 多轮 control memory

T2.1.2 的 `physics/control_memory.py` 只接收 T2.1.1 `ObservedSyndromeStep` 与本周期实际执行的 `ControlDecision`。它以此前 post-action estimate 为参考做 nearest periodic lift，随后按现有 `LogicalErrorTracker/FastLoopEmulator` 约定减去 correction；同时更新 confidence、GF(2) Pauli frame、modulo-2pi phase frame、分离 e/leakage runs、单调 parameter-bank version 和 deadline current/run/count。

full `SyndromeStreamStep` 会被类型拒绝，防止 hidden truth 错接 controller。deadline miss 只记录事实，不自动清零 correction，因为 local safe fallback 仍可能执行实际动作；deadline enforcement、fallback FSM、CRC/age/CAS/ack 和 confidence calibration 均尚未实现。详细 contract 与 26 项 direct tests 见 `docs/multiround_control_memory.md`。

### 3.8 Fast Monte Carlo 与 rare-event strata

T2.1.3 的 `physics/fast_monte_carlo.py` 按 round 循环并在 independent trajectories 维度向量化，保留 residual、recovery depth/axis、persistent leakage 与 logical parity state。production run 完成 10,000×100=1,000,000 cycles，host 耗时 `0.2531 s`；weighted `P_L=0.04702085`，95% trajectory-cluster bootstrap CI `[0.04643899,0.04756083]`。

rare mode 把“是否存在一个额外 burst/leakage episode”定义为已知 probability 的 trajectory mixture；normal/rare conditional rates 按 target weights 合成，而不是抄过采样 raw fraction。确定性 burst/leakage tests 与 allocation invariance 已证明 estimator contract。所有 rare probability/duration/scale 仍是 scenario assumptions；host cycles/s 也不是 target-board timing。详细见 `docs/fast_monte_carlo_validation.md` 和机器 JSON。

### 3.9 Finite-squeezing 分解式 effective noise

T2.2.1 的 `physics/finite_squeezing_noise.py` 不使用单个 `sigma_eff`。physical truth 明确写成 channel + data-GKP peak + finite-energy envelope；observed analog 再加 ancilla-GKP peak 与 classical measurement。canonical damped-projector probability variance 是 `tanh(Delta^2)/2`；本模型 decoder-standardized lanes 使用 `tanh(Delta^2)`，data/ancilla 仍进入不同 schema lane。

envelope 项由 isolated-peak lattice-index weights 与 `sech(Delta^2)` center contraction 生成离散 non-Gaussian shifts，不是另加一个 Gaussian。25 万样本×6 点 sweep 的 finite-squeezing excess trace 从 `0.327704` 严格降到 0，最大经验/解析 observed-covariance 相对误差 `0.004554`；broad/ideal `P_L` 分别为 `0.031260 [0.030585,0.031949]` 与 `0.000424 [0.000351,0.000513]`。`Delta=0` 时 finite-squeezing excess 逐元素为零，physical/observed 精确退化为 channel/channel+measurement。该 isolated-peak effective mixture 不是 coherent Fock state，Delta 也未映射为装置 squeezing dB。详细见 `docs/finite_squeezing_effective_model.md` 和机器 JSON。

### 3.10 协议原生 ancilla/readout/reset fault flow

T2.2.2 的 `physics/protocol_ancilla_errors.py` 在既有 sBs 4×3 hidden/observed/reset kernel 上叠加 X/Z constituent × small-CD/big-CD/readout 的 bit/phase fault。bit parity 可切换 constituent outcome；big-CD bit fault 可产生 hidden logical backaction；phase fault不翻转 Z-basis outcome。misclassification 后的 virtual-rotation error、fault stage 和 physical backaction 只存在 truth schema，deployable record 仍只有 observed syndrome/reset/run fields。

sharpen--trim 不复用 sBs `g/e/leakage` 字母表，而是独立运行 q-peak、p-peak、q-trim、p-trim 四轮 `+y/-y` 状态机。3×2 full confusion 把 hidden `+y/-y/leakage` 映射为二值观测，reset failure/leakage 可跨轮 carry，但 hidden counters 不进入 deployable view；peak middle-half bit fault 的 stochastic logical backaction也不会泄漏到 deterministic Pauli frame。80,000 sBs cycles 与 80,000 sharpen--trim rounds 的 6 项统计/非执行门全 PASS。详见 `docs/protocol_ancilla_errors.md` 和机器 JSON。

### 3.11 控制链与 active-correction imperfection

T2.2.3 的 `physics/control_imperfections.py` 把 requested correction、controller-visible AWG/DAC codes 和 simulator-only physical displacement 分开。Cartesian command 先经过 unsigned amplitude/periodic phase AWG quantization，再经过 signed DAC I/Q quantization；随后应用 affine pulse gain/crosstalk/bias、command-dependent multiplicative error、full-covariance additive error、latency drift/diffusion 和独立 virtual-rotation quantization/calibration/noise。两种 noncommuting action order 都有 exact first/second moments。

100,000-sample production 的最大 mean z-score 为 `1.5358`，covariance relative error 为 `0.001697`；12 radii×73 phases 的 6/8/10/12-bit RMS error 从 `0.041179` 严格降到 `0.000627`，latency-only covariance trace 随 0/2/5/10 us 严格增长，ideal endpoint residual 精确为零。该层已接现有 Q4.20 fast-loop，但 bit depth、pulse matrix 与 latency law 均为 assumptions，不是目标板或微波链实测。详见 `docs/control_imperfections.md` 和机器 JSON。

### 3.12 Generic finite-Fock density reference

T2.3.1 的 `physics/fock_density_model.py` 不调用旧 heuristic signed-grid fallback。它把
T1.2.1 已归一化的 damped-projector position wavefunction 投影到递推 Hermite basis，
保留 cutoff 重归一化前的 captured probability；随后以 density matrix 显式实现
displacement、pure-loss Kraus、sparse thermal Lindblad、Gaussian phase diffusion、Kerr
和 modular-displacement POVM 的条件/非选择 backaction。

`Delta=0.45` 的 `N=18/24/30/36` capture 从 `0.999729` 增至 `0.99999961`，最后相邻
嵌入 fidelity 为 `0.99999882`；loss、thermal、phase、Kerr 与 measurement analytic errors
均保持数值精度，direct tests 与 10 个收紧后的 production gates 全通过。high-Fock
upward-shift 只是一个 CPTP cavity-occupation stressor，不是 transmon `|f>` leakage。
该 generic reference 本身不执行完整协议一轮；T2.3.2 在独立模块中复用其 operator/channel
能力，避免把 generic channel API 和 protocol semantics 混成同一层。详见
`docs/fock_density_model.md` 和机器 JSON。

### 3.13 Finite-cutoff completed analytic SBS one-round

T2.3.2 的 `physics/fock_sbs_cycle.py` 直接实现一手补充材料的 `K_g/e^X`，并以
`(x,p)->(-p,x)` 生成 `K_g/e^Z`；不再使用 modular measurement + artificial feedback
displacement + ideal logical lift 的替代拼装。仓库 decoder q 到论文 canonical x 的
`sqrt(2)` dilation 已完整覆盖 center、width、envelope、Jacobian，并进入显式 API 与回归。

有限 Fock 边界会破坏 raw Kraus completeness。production `N=24,Delta=0.34` 的 X/Z
raw full-space error 为 `1.091934`、code-subspace error 均为 `0.016552`；实现保留
该 defect 与 Gram 谱，再用共享 `K_b G^(-1/2)` completion 把误差降至约 `1e-14`。
因此证据名固定为 finite-cutoff completed analytic SBS map，不升级成 infinite-dimensional
exact channel。

一轮覆盖 initialization、idle、hidden Kraus、noisy observed routing、scenario residual
classical phase action、deterministic X/Z Pauli frame 和 logical projection。perfect readout
产生四条物理 Kraus path，非零 confusion 产生 16 条 hidden/observed branch。六逻辑态
clean conditional fidelity/survival 为 `0.999953/0.969508`，`a/a^dagger` 首层误差的
单轮平均 code-survival gain 为 `0.518107`；100k MC、五点 cutoff、direct 与 16 个
production gates 全通过。详见 `docs/fock_sbs_cycle.md` 和机器 JSON。

### 3.14 Signal/noise/logical-jump noise-transfer surrogate

T2.3.8 的 `physics/noise_transfer_surrogate.py` 不复用 T2.2.1 sampler 冒充新 lane。
它分别传播整数 lattice signal、连续 offset/covariance 与 nearest-domain alias parity：loss
包含 signal-index-dependent attenuation bias 和 vacuum injection，measurement efficiency
写成 equivalent covariance，任意 2×2 feedforward gain 决定 output covariance。

每轴 Gaussian cell probability、folded moments 与 domain-conditioned clipping ratio 均为
解析量。diagonal decision covariance 才给 exact I/X/Z/Y product law；相关 q-p 时只给边缘
概率与 Fréchet bounds。stored axes 是 decoder-standardized classical axes，vacuum variance
为 `1`；Fock 对照转为 canonical。10/12 dB normalized state/Fock moments 对齐通过；3 dB
出现 `56.04%` proxy error 与 `37.33%` 四逻辑态 spread，被明确判为 clipping-dominated
失效区。45 direct tests、独立 quadrature、200k MC 与 14 个 production gates 全通过。
详见 `docs/noise_transfer_surrogate.md` 和机器 JSON。该层不是 paper-exact teleportation、
SBS Kraus 或 device model。

### 3.15 四 lane axis-resolved cross-fidelity

T2.3.3 用同一 `3/5/8/10/12 dB`、channel/measurement/ancilla-peak contract 运行
finite-Fock folded density、T2.2.1 stochastic effective、T2.3.8 exact alias 和 normalized
state-density syndrome MAP。Fock q/p residual-parity 都在 canonical chart 计算，经显式
independent-axis projection 生成 Pauli-twirled LER/occupancy/`F_avg`；Fock SBS code survival 等 native metrics 分列，
不与 central-domain mass 做绝对排名。

四 lane 的共同 LER 均严格下降，occupancy/`F_avg` 均严格上升。10/12 dB
noise-transfer↔syndrome q-LER 最大差 `3.93e-5`，effective 200k MC 最大 z `1.708`；
3 dB gap `0.015408` 与 clipping ratio `0.358` 作为失效区。12 dB Fock `N=24→48`
q-LER 从 `0.011009` 降到 `0.002323`，因此只通过 cutoff-aware absolute gate。

T-RISK-20260714-01 修复后，10/12 dB canonical q/p LER 最大差 `1.51e-7`；旧含混
operational Fourier 路径仍有最小 `0.418` 差并作为负证据保留。axis-resolved 对齐不等于
joint coherent q-p correlation/process fidelity，后者仍 fail closed。详见
`docs/cross_fidelity_validation.md` 和机器 JSON。

### 3.16 Quadrature normalization contract

`physics/quadrature_conventions.py` 分开 canonical Fock、decoder-standardized、Sivak
displacement-amplitude 与 anisotropic symplectic bridge。decoder 两轴同乘 `sqrt(2)` 的
determinant 为 2，作为 joint operator pair 会把 commutator 从 `i` 改成 `2i`，API 在
quantum-symplectic 路径主动拒绝；`diag(sqrt(2),1/sqrt(2))` 才保持 determinant 1。
15 个机器 gate 覆盖 reciprocal lattice、wavefunction Jacobian、vector/covariance/peak
roundtrip、FFT q/p moments 和 alias invariance。详见 `docs/quadrature_normalization_contract.md`。

### 3.17 Joint cavity--ancilla differentiable trajectory

T2.3.4 的 `physics/differentiable_sbs_trajectory.py` 不用固定参数 analytic Kraus 外壳
冒充 Feedback-GRAPE。它在 `2N` joint density 上显式执行四个 qubit rotations、三个
complex ECD、固定 layer-4 displacement 和 VR；前 14 个 correction 受限于 `[-2,2]`，
VR 受限于 `[-1,1]`。每个 half-cycle 的七段 idle 使用 exact finite-cutoff cavity loss、
two-level ancilla amplitude damping/dephasing Kraus，随后按 Born probability 采样/回放
`g/e`、reset ancilla 并累积 `log P_theta(m)`。

causal `control_policy(history,j)` 在第 `j` 步只能读取此前 outcome；sampling decision
detach，但 selected branch 的 reward 与 log-probability graph 不 detach。cutoff-8 CPU/CUDA
各 17 个 machine gates 和 37 direct tests 通过；四条 two-measurement branch 和为 `1.0`，
open-loop/history gradient norm 为 `0.9365/1.0424`，gate/CPTP residual 不超过
`8.89e-15/8.89e-16`。这里的 `5 us` half-cycle 是 Puviani Table S1 数值 profile，不替代
Sivak Table S3 `4.924 us` constituent，也不是硬件测量。详见
`docs/differentiable_sbs_trajectory.md` 与 CPU/CUDA machine JSON。

### 3.18 Feedback-GRAPE reward/score gradient validation

T2.3.5 的 `physics/feedback_grape_gradient.py` 对一轮四条 measurement trajectory 做
完整穷举，分别计算 conditional reward path
`E[dR/dtheta]` 与 likelihood-score path `E[R d log(P)/dtheta]`，而不是只验证总 loss
能反传。cutoff-6 的三参数 causal audit policy 得到 exact gradient
`[0.18949777,-0.07522291,0.00863952]`；两项之和与直接微分
`sum_m P_m R_m` 的最大差为 `5.55e-17`。常数 baseline 的 exact invariance 残差为
`1.11e-16`，`E[d log(P)/dtheta]=0` 的残差为 `1.51e-16`。

总梯度、reward path 与 score path 分别冻结其互补因子做 central finite difference；
`h=1e-5` 的 relative L2 error 为 `1.68e-10/2.44e-10/3.22e-10`，四点步长
sweep 的最差误差均低于 `2.6e-7`。32 个独立 batch、每批 384 条、合计 12,288
条随机轨迹的 component z-score 最大为 `1.120`，constant baseline 将 score trace
variance 降到原来的 `0.04443`。两轮十六分支、CUDA exact parity 与 32 项 direct tests
另作防伪检查。这里的 compact policy 只是梯度审计探针，不等于 RNN teacher、训练收敛
或可行资源包络。详见 `docs/feedback_grape_gradient_validation.md` 与机器 JSON。

### 3.19 Current-host recurrent Adam resource envelope

T2.3.6 不复用 T2.3.4 的 forward-only resource counter。每个点实际运行
GRU10--256--256--15 causal policy、随机 joint trajectory、reward/score backward 与
`Adam(lr=1e-4)` parameter update；1 次 warm-up 后保留 3 次 raw step time。56 个隔离 CUDA
点与 9 个 CPU 点覆盖 cutoff `8--48`、batch `1--576`，cutoff 16 的 batch 8/16 覆盖每个
2--10 full-cycle horizon。CUDA peak allocated/reserved 包含 autograd/Adam state，CPU 用
2 ms sampler 测 RSS。

cutoff 16、batch 16 的 2--10-cycle median 为 `0.208--1.050 s`，peak allocation
`83.2--303.8 MB`。batch 512、10 cycles 以 `8.624 s/6.233 GB (72.60%)` 通过；batch
576 以 `7.015 GB (81.71%)` 触发 memory gate。cutoff 48、batch 8、10 cycles 通过，
batch 16 以 `13.954 s` 触发 runtime gate。65 点均保持 finite gradient/update，最大 trace
error `6.66e-16`、最小本征值 `-9.37e-16`。这证明的是当前 host training kernel
feasibility，不是 optimizer convergence、physical cutoff convergence、NMF gain 或硬件
timing。详见 `docs/differentiable_sbs_feasibility.md`、机器 JSON/CSV 与 figure bundle。

### 3.20 Strict-split MF/NMF directional ranking

T2.3.7 的 `physics/nmf_directional_ranking.py` 使用相同 finite-cutoff joint
cavity--two-level-ancilla simulator、Puviani Table S1 high-noise timing、10-cycle 物理时间和
noise contract 比较 standard、70,159 参数 latest-outcome MF 与 72,853 参数 full-history
NMF。5 个 train/agent seeds、2 个 validation seeds、8 个 primary test seeds 和 4 个
cutoff-confirmation seeds 严格分离；score baseline 只使用 train-only warm-up，validation
只选 checkpoint，test 不参与 agent post-selection。schema-v3 checkpoint 同时绑定 config、
executable source bytes 与 training protocol，并逐模型复核 SHA-256。

cutoff 12 的 projected-logical-Z area-equivalent lifetime 为
`standard/MF/NMF = 2.7477/6.5347/6.7408 cycles`；5/5 配对 NMF agents 高于 MF，
NMF−MF 20,000 次 bootstrap 95% CI 为 `[0.0842,0.3281]` cycles。physical fidelity 与
logical-Z AUC 保持同方向；cutoff 16 独立 lane 仍为
`NMF 7.7084 > MF 7.2459 > standard 5.1442`。primary hidden-reset ablation 降到
`6.0317`，但 cutoff 16 reset view 反而高于 full-history NMF，因此完整 history gain 尚未
跨 cutoff 稳健，不能升级为长期 memory mechanism claim。

本门只支持 finite-cutoff two-level 10-cycle directional ranking。300-epoch NMF 的 4/5
best checkpoint 位于末轮，optimizer convergence、论文 1000-cycle 六态 channel lifetime、
论文增益幅度、pulse/leakage/SPAM/device/hardware 仍 fail closed。详见
`docs/nmf_directional_ranking.md`、schema-v3 JSON/checkpoint、8,450-row Source Data 与
publication figure bundle。

### 3.21 Exact-budget latest-outcome Markovian comparator

T3.2.7 不直接把旧 70,159 参数 MF 当作 memory-specific 最终对照，而是在
`physics/latest_outcome_markovian.py` 中以 390 参数/330 dense-MAC static front 精确替换
GRU front；两者总参数/MAC 同为 `72,853/72,266`，共享 15 维 action、5-agent/300-epoch
Feedback-GRAPE protocol 和 held-out traces。learned path 只读取最新 g/e/leakage token，无
hidden state；当前 two-level production lane 的 leakage count 为零，不能声称 leakage robustness。

cutoff 12 的 `exact MF/history NMF = 6.888249/6.740785 cycles`，配对
`NMF-exact=-0.147464 [-0.386866,0.147532]`；cutoff 16 则变为
`7.168269/7.708351`，五 agent 同法 bootstrap 为 `+0.540082
[0.231972,0.785521]`。因此旧 `NMF > MF` 排名不足以单独证明 memory mechanism；必须保留
cutoff-dependent direction reversal，并交给 T3.2.11/T5.4.5 做多 cutoff history 消融和长时验证。

### 3.22 Autonomous sBs wall-clock baseline

T3.2.8 不把 autonomous 的 `0.7 standard cycle` 当作 lifetime 缩放因子，而是在
`physics/autonomous_sbs.py` 中分别执行 measurement-feedback 10 us 与 autonomous 7 us
full-cycle channel，推进到共同 700 us。两条路径复用相同显式 joint cavity--ancilla gates；
measurement-feedback 做 nonselective outcome sum，autonomous 省略 measurement event 但仍
trace/reset ancilla，并分别按原生 duration 施加 decoherence。

cutoff `12/16` 与 Table S5 high/medium/low 六条 lane 中，按各自 protocol cycle 的
autonomous/measurement lifetime 比为 `1.151287--1.346101`，按共同物理时间却为
`0.805901--0.942271`。700 us 下前者执行 100 cycles、0 measurements、200 resets、1,800
active gates；后者为 70 cycles、140 measurements、140 resets、1,260 active gates。机器
artifact 同时保存两种单位和 raw event counts，并以四分支枚举等价与 zero-noise duration
invariance 防止把简化缩放冒充协议仿真。

该结果仅是 finite-cutoff two-level、nominal-control、literature-timing model baseline。
autonomous-specific pulse/control optimization、Fig. 3(b) 数值复现、multilevel leakage、
device/target-board timing 均未完成；cutoff 12 的末端单点与全时域 area lifetime 还会给出
相反表面排序，故禁止只报 endpoint 或 per-cycle 数字。

### 3.23 Finite-horizon trajectory lookup control reference

T3.2.9 在 measurement-feedback sBs 的两个 full cycles/四个 g/e outcomes 上建立完整因果
prefix tree。第 j 个 half-cycle action 只读 `[0,j)` history；节点数 `1+2+4+8=15`，每节点
15 个 bounded residual，共 225 scalars，终端 16 branches。全部 branch 同时进入
`sum P_theta(m) * fidelity(m)`，所以早期 action 对 trajectory probability 的影响保留；不做
逐条完整轨迹的 hindsight action。

为分离 history 与普通 retuning，另优化 4×15 time-indexed open-loop，并把最佳表按深度精确
复制成 lookup warm start。三 restart 各经 300-epoch phase-one 与 250-epoch low-rate
refinement 后，cutoff12 standard/open/lookup fidelity 为
`0.396787/0.769403/0.815799`；冻结表在 cutoff16 为
`0.559221/0.503415/0.638688`。lookup 的 history-specific margin 为 `+0.046396`，但 terminal
probability skew 到 `7.57e-5--0.6230`，p(g) 比 standard 低 `0.10065`；open-loop cutoff
排序还发生反转。

因此 `finite_horizon_control_oracle` 在 registry 中只升级为 empirical multi-start software
reference，不是 globally certified optimum。资源按 `2^(2C)` 增长：10 cycles 已有
1,048,575 action nodes、15,728,625 scalars 和约 9 GiB terminal-density lower bound，明确
nondeployable；它也不是 decoder oracle、channel-recovery bound、Fig. S4 数值复现或硬件结果。

### 3.24 PRL-inspired exponential-recurrence control baseline

T3.2.10 把 Puviani Supplement 的 `pi[t+1]=a[m]pi[t]+(1-a[m])pi_inf[m]` 固定成
15-control 因果策略。物理 lane 在 two-level measurement-feedback sBs 上 exact 枚举两 cycles
的 16 个 `g/e` branches，训练 75 scalars；leakage branch 仅固定接口，不冒充物理训练。
3 restart 各 `300+250` epochs 后，cutoff12 standard/recurrence/lookup fidelity 为
`0.396787/0.784921/0.815799`，software Q mirror 为 `0.784921`。

cutoff16 只冻结迁移，recurrence/lookup 为 `0.773930/0.638688`。该反转不能写成超越 oracle：
lookup 是 cutoff12 非凸 multi-start empirical reference，跨 cutoff 不是全局上界。recurrence 的
`p(g)=0.984661` 还表明 gain 含 branch reshaping。独立 synthetic event-cost lane 在同 384k
traces 上得到 recurrence/FSM/memoryless `0.073618/0.202829/0.022917`；它只证明此 cost
matrix 下的排序，不是 physical fidelity/LER。Q4.16/Q2.18 mode parity `99.9716%` 仍只是
software integer evidence，不是 RTL、综合或板测。

### 3.25 Memory-specific causal interventions

T3.2.11 对 T2.3.7 五个冻结 NMF parent 执行 prefix-consistent history shuffle、sliding
truncation、periodic hidden reset 和 frozen latest-only，并与 T3.2.7 五个 independently
retrained、同 72,853 参数/72,266 MAC 的 latest-only FNN 比较。所有 intervention 只重放
observed prefix；full-view 与 parent bit-exact，weights 不变，每个变体都重新闭环推进物理
trajectory。

cutoff12 full/retrained/frozen-latest logical-Z lifetime 为
`6.740785/6.888249/6.031675`，cutoff16 则为 `7.708351/7.168269/8.271987`。
truncation/reset 的 memory-length 方向也在两个 cutoff 反转。三组稳定 prefix shuffle 的
full-minus-shuffle 为 cutoff12 `+0.068617 [0.050734,0.084185]`，cutoff16
`+0.046558 [-0.014033,0.107150]`。因此预注册四对照联合规则在两 lane 均失败，只允许写
`cross_cutoff_memory_mechanism_not_supported`；不能外推 universal memory gain/loss、论文
1000-cycle mechanism、multilevel leakage 或硬件结论。

### 3.26 Matched-budget slow-loop model-family selection

T4.1.1 不拼接不同 T3 物理 fidelity/event cost，而是统一复用 T3.2.6 four-regime
posterior task：每 32 cycles 形成 14 项 observed summary，所有 family 只看最近 8 windows，
共同受 4096 MAC、4096 B 常驻模型/状态和 4096 B transient workspace 约束。3 training、
3 validation、8 evaluation seeds 完全分离，validation NLL 决定 winner，evaluation 不重选。

validation HMM/TCN/GRU NLL 为 `0.454975/0.476180/0.503134`；evaluation 为
`0.455711/0.511936/0.509619`。validation runner-up-minus-HMM 的 8-seed evaluation CI
为 `[0.046709,0.065742]`。HMM 用 exact rolling emission cache 达到 926 MAC、3728 B
常驻、104 B scratch，并与朴素 last-8 replay 在 `1e-13` 内一致。该结果只冻结 synthetic
pilot backbone；HMM detection delay、richer T4.1.2 input、OOD/device/fixed-point/FPGA 仍需
后续任务，禁止写 universal architecture superiority。

### 3.27 Observed-only experimental history contract

T4.1.2 在 cycle 末端连接 `ObservedSyndromeStep`、`RunLengthFSMDecision`、registered-calibration
`llr_1d` 和 `DualLoopScheduler` event/snapshot，形成 256×53 slow-loop history。10 个 feature groups
覆盖 analog/residual syndrome、X/Z `g/e/leakage` one-hot、phase sin/cos、same-cycle applied action、
LLR、run counters、deadline/communication/window age、六类 update status、bank state 与 record health。
左 padding 另带 mask/cycle index；LLR/run/version/queue clipping 都有 saturation flag。

该 schema 只在 cycle `t` 完成后预测未来慢状态，不允许 hindsight action。字段名、字符串值、
nested object 与 scheduler payload 全部走 truth-leak denylist，`DriftState`/`SyndromeTruthStep` 直接
拒绝。8 seeds×2,048 cycles 的真实 repository-producer replay 通过 17 gates，覆盖 6 update statuses、
5 FSM modes、g/e/leakage、deadline/通信/CRC/failure/stale/conflict/commit/FIFO/saturation。它仍是
synthetic software input contract；不是 raw IQ/ADC/device calibration、richer-input learned-model gain、
RTL 或 board timing 证据。

### 3.28 Future-only hybrid slow-state output

T4.1.3 在 256-cycle history 末端输出 9 个 continuous observed-noise/calibration 参数、四态
regime posterior、next/horizon leakage risk、0--6 estimated recovery-burden posterior、9×9
moving-block-bootstrap uncertainty，以及带 base version/validity/ID/CRC32 的 inactive-bank K/b proposal。
T4.1.1 Gaussian HMM checkpoint 按 manifest hash 恢复，但仍消费原 14-summary×8-window path；这只是
兼容桥，不是 53-feature retraining。

8 seeds×2,048 cycles 的 nominal/stress replay 产生 456 outputs；58 次 stage 在下一 cycle 精确
atomic commit，398 次 uncertainty/fault gate hold，五种 profile 均覆盖。payload 不含逐周期
correction/frame/pulse action；recovery 是 observed burden posterior，不是 hidden depth。risk/calibration、
OOD、logical/control gain、fixed-point/RTL/board/device 仍需 T4.1.4/T4.2/T5/T6，禁止从 schema/
atomicity 反推性能。

### 3.29 Strict-split multi-objective calibration

T4.1.4 将 T4.1.3 outputs 与下一 32 cycles 对齐，冻结 state estimation、paired oracle-gap、regime
NLL+Brier、Gaussian uncertainty log score、false fallback 和 slow update cost 六项目标。3/2/3 seed
training/validation/evaluation 完全隔离：training 只产生 robust scales，validation 只选 posterior
temperature/uniform mix、uncertainty scale 和 safety-constrained threshold，evaluation 不参与选择。

448 条 future-aligned records 通过 19 gates；validation regime NLL 和 95% coverage 从
`8.563501/0.479167` 校准到 `1.262520/0.947421`。但 evaluation 在 unsafe recall `1.0` 时
false-fallback 也为 `1.0`，regime accuracy 也未提升。这是旧 HMM bridge/risk score 缺选择性的负结果，
不是 deployable calibration。offline truth 只存在于 evaluator Source Data；不得进入 history/output/fast path。

### 3.30 Offline teacher / online recurrence student

T4.1.5 从 T2.3.7 checkpoint/manifest hash 恢复 5 个 frozen NMF policy，只允许其离线产生 15-control raw
residual target。training/validation/evaluation 各 256 条 20-half-cycle trajectory，seed 与 hash 完全隔离；
3 restart 的 75 参数指数递推由 validation-only 选择，在线 artifact 共 105 scalars/420 float32 bytes。

冻结 evaluation student/latest-only/zero-safe imitation MSE 为
`1.453624e-6/1.404389e-4/5.265504e-3`。在线模块只接收 observed `g/e` 和 health flags；leakage、invalid、
CRC failure、stale parameter 或 deadline miss 均输出零 raw residual，teacher/torch/simulator 不进入 runtime。
这里的 retention 仅是 teacher-action MSE 定义，不能替代 lifetime/fidelity/`p(g)` 物理同轨验证，也不是
leakage-trained、fixed-point、RTL、FPGA timing 或 board/device 证据。

### 3.31 Version-bound axis-marginal parametric MAP-LUT

T4.2.1 从真实 ParamBank active `K/b` 和 measurement covariance 反解 effective mean/covariance，重新验证
`K=C(C+R)^-1` 后在 slow path 编译 X/Z 两张 Q9.12 LLR ROM。fast path 只接收 10-bit syndrome code、
phase bit 与 latched version，执行 coarse address、guard-node ROM read、half-bin-corrected integer interpolation
和 sign action；没有在线浮点 likelihood、除法或 `exp/log`。

8 banks×2 phases×1024 codes 的 16,384-row exhaustive replay hard action mismatch 为 0，mean/max LLR
code error 为 `0.387756/20`；5--8 address-bit error 严格下降。五级 software pipeline contract 为 latency
5 cycles、II=1，并锁存 in-flight image/version。它只证明 independent phase 的 marginal MAP executor；不证明
full correlated 2D optimality、event/frame/fallback 集成、RTL/synthesis/FPGA timing 或 board/device measurement。

### 3.32 Observed-event FSM 与 frame action

T4.2.2 将 T4.2.1 对齐后的 MAP decision/version 接到独立六态 FSM；输入只含 X/Z `g/e/leakage`、
phase、reset ack 与 valid/CRC/fresh/deadline health。六个 3-bit counter 均饱和不回绕，priority 覆盖
health fail-closed、sticky reset、leakage hold/reset、hold/fallback hysteresis、X/Z recovery 与同时 `e`
的 phase tie-break。安全态禁用 correction、抑制 pending flip 且 frame delta 保持零。

非安全态的 MAP flip 原子更新 GF(2) Pauli frame 和 8-bit modulo phase-frame；后者只是逻辑表示镜像，
不是物理微波相位。8×128-cycle replay 的 20 gates 覆盖六态/transition/counter saturation、reset/fault、
8-bank hash/version、transactional negatives 与 deterministic replay。T4.2.1 五周期 MAP 加一周期 action register
得到 6-cycle/II=1 software contract；完整 conservative fallback、定点 LER、RTL/综合/FPGA/board 仍未完成。

### 3.33 Traceable conservative health/fallback

T4.2.3 在 event FSM 前增加 controller-owned trusted image registry 和 14-bit health/integrity taxonomy，覆盖
observation/OOD、input/image CRC/SHA、unknown/mismatch/rollback version、parameter age、deadline、MAP missing/
alignment/action、unexpected ack 与 leakage。blocking fault 不消费 MAP、不接受新 version，Pauli/phase-frame
delta 全为零，并输出固定 bitmask、primary/full reason trace 和 uint8 saturating per-reason counters。

16×256-cycle replay 的 20 gates 验证 OOD 192/193 与 age 64/65 边界、8-bank monotonic commit、rollback/
unknown/mismatch 保持 last trusted version、leakage reset/ack、组合 fault、两周期恢复、counter saturation、
6-cycle/II=1 和 deterministic replay。fallback 是 `frame_hold_no_map`，不是自动 bank-0 rollback；OOD 校准、
片上 CRC 重算、transport watchdog、定点 LER、RTL/综合/FPGA/board 仍未完成。

### 3.34 End-to-end bit-accurate MAP--health--event--frame path

T4.2.4 将 float replay encoder 与 integer online `step_codes` 分离，冻结 selected 10-bit ADC、8+2-bit
address/fraction、signed Q9.12 LLR、六态/六个 3-bit event counters、2×8-bit phase frame、8-bit OOD、
16-bit age/version、14-bit fault mask、18×8-bit health counters和 CRC32/SHA256 exact comparison。compile/interpolate
均 nearest-even + saturation；counter 不回绕，Pauli/phase frame 分别 XOR/modulo。

四档 precision×8 banks×X/Z 完整 audit 共 87,040 codes；selected hard mismatch 0，mean/max LLR value error
`9.46671e-5/0.00488281`。8 banks×4 seeds×2,048 的 paired LER 中 selected quantized-minus-float 为
`3.05176e-5 [-4.57764e-5,1.22070e-4]`，CI 跨零。完整 Source Data 独立重编译/重跑 hash 一致。这里仍是
model-matched axis-marginal software reference；correlated/OOD/fault LER、综合资源、FPGA/board 尚未完成。

### 3.35 Three-timescale cadence and adaptation lag

T4.3.1 将 T4.2 integer fast path 接到真实 `DualLoopScheduler`/`ParamBank`：fast 配置周期 5 us，local urgent
event 经 1-cycle register 在下一边界动作；最近 2048 个 valid samples 形成 10.24 ms 内容窗口，但每 4000
cycles/20 ms 才发窗并同相启动 slow job。当前 latency-model 均值服务为 995 us/199 cycles，完成周期 stage，
下一 cycle commit，且 commit 先于该 cycle fast callback。60 s 和 end-of-run 只发 host recalibration-due ticket。

全 4000 onset phases 下，首个受影响窗口的 onset-to-first-use 为 `1.000--20.995 ms`，完整 post-change 窗口为
`11.235--31.230 ms`；local event 始终为 5 us。两种 lag 不得混称。集成审计还把生产 max parameter age 从
独立 pilot 的 64-cycle 假设分离，HIL cadence 冻结 8192 cycles，允许一个 20 ms update 缺口后继续，第二个
缺口后 fail closed。以上均是 software/config reference；jitter/queue/pause/race、完整 CRC/CAS/readback/
hysteresis、RTL/FPGA/board/device 仍未完成。

### 3.36 Atomic parameter-image bank and hysteresis

T4.3.2 将完整 T4.2 parametric MAP-LUT image 作为事务单位。partial transfer 保存在 A/B valid slots 之外；只有 manifest CRC/SHA、payload CRC/SHA、canonical decode、image self-check、version/CAS、cycle timestamp、source-window 和两窗同 key hysteresis 全部通过，完整 image 才一次发布到 inactive slot。commit 还需到达 apply epoch、显式 safe cycle boundary、4000-cycle minimum residency，并再次检查 CAS/freshness/hysteresis；成功后只交换 active pointer，host 用 ack/readback 的 bank/version/activation epoch/image CRC/SHA 确认。

3745 个 proper prefix、3745 个单字节翻转、10 个 chunk/order cases、15 个 semantic negatives 和双 bank/pipeline/race 集成共 7518 行 Source Data，17/17 gates 通过；A:v0→B:v1→A:v2，旧 in-flight request 输出 v0，新 request 输出 v1。这仍是 thread-safe software transaction contract；旧 scheduler 未迁移，caller-provided safe boundary、automatic rollback、transport、CDC/RTL/FPGA/board/device 尚未完成。

### 3.37 Fault recovery and monotonic LKG republish

T4.3.3 将 T4.3.2 atomic image bank 与 T4.2 bit-accurate fallback 接成逐周期 software closed loop。commit ack 丢失时 host 保持 uncertain，阻止后续 writer，通信恢复后才用 readback 确认；host timeout/parameter stale、OOD/deadline、image integrity 和 leakage 分别落到 frame hold 或 reset request，不让 blocking fault 延用 correction。

post-commit guard 失败不降低 active version，而将 prior LKG contents 作为新完整 image 单调重发，例如 `A:v0 -> B:v1 -> A:v2(LKG contents)`。8 scenarios×4 seeds 共 767872 cycles 的 undefined action、blocking-fault correction、frame overflow 均为 0；周期 refresh 又关闭了“host 在线但 image 超龄”的首轮缺口。这仍是 software recovery contract；policy calibration、wire/CDC/RTL/FPGA/board 和物理稳定性未完成。

### 3.38 Fresh bounded-residual GRU teacher

T4.4.1 复用 T2.3.7 的 simulator/Feedback-GRAPE 实现和 T3.2.9 的
`SBS-NOMINAL-PLUS-BOUNDED-RESIDUAL-15` 合同，但用与旧 teacher 完全不交的 `601/709/811` 三个 seeds
重新初始化并训练 GRU10-Dense256-Dense256-Out15。动作始终为
`nominal + [2×14,1] * tanh(raw)`；zero residual 严格等于 nominal，全零物理门向量没有被冒充安全 initializer。

三个 restart 的 validation score 分别从约 `0.303` 升至 `0.588/0.585/0.587`，validation-only 选择
restart 601。cutoff12/16 held-out score 相对 nominal 增益为 `0.253603/0.141557`，primary logical-Z
lifetime paired gain 为 `4.125518 [3.899115,4.339331]` cycles。21 gates、1,074-row Source Data、全部参数
gradient、causal/cached replay 和 checkpoint/hash reload 通过。restart 601/811 在 epoch 320 达峰并显式记为
budget-cap hit；因此这里只证明 fresh finite-model teacher 可训练，不证明 optimizer 全局收敛、paper-exact
long-horizon、multilevel leakage/SPAM/pulse、online student、FPGA 或 device gain。

### 3.39 Frozen-teacher hidden/control response geometry

T4.4.2 不更新 T4.4.1 teacher 权重，而对 10 条 native `g/e` fixed sequences 做 128-half-cycle hidden/control
追踪，并对前 20 half-cycles 做 forced-path conditional `p(g)` 仿真。另用 24/8 trajectory-disjoint split 比较
10-D hidden linear probe 与 5-D observed-history probe；hidden evaluation `R²=0.667797`，比简单 observed
features 高 `0.180404`，但 target 仍是 assumed-model `p(g)`，不是 device-calibrated physical belief。

hidden/control 的 95% variance 都集中在 1 PC，hidden 99% 需要 2 PC；30 个 all-g/all-e 逐参数单指数拟合中
28 个 `R²>=0.95`，但 virtual rotation 的 all-g/all-e 为 `0.696/0.935`，所以“全部参数单指数”被明确否证。
双向 impulse control response 在 10/12 half-cycles 降到峰值 1%，g/e fixed-point Jacobian radius 为
`0.618/0.596`。leakage 不作为第三 token 输入 teacher，只登记 reset-hidden + first-post-leak nominal action 的
OOD proxy。以上只为 T4.4.3 student 结构候选，不能替代 T4.4.4 gain-retention、leakage/OOD/long-horizon 或硬件门。

### 3.40 Strict-split low-dimensional exponential student

T4.4.3 新建 1/2/4-state outcome-specific exponential recurrences，每个维度保留 3 个 fresh 900-epoch
restarts。training/validation/evaluation 各 256×64 histories 严格分离，validation 先选每维 restart，再按冻结
`5% + 1e-7` tolerance 选择最小合格维度；evaluation 不参与选模。唯一合格的是 4-state restart 0，其
validation/evaluation MSE 为 `5.648504e-6/6.083136e-6`。

selected student 为 95 scalars、4 persistent states、87 analytic MAC/healthy step；evaluation MSE 相对
zero/latest-only/legacy T4.1.5 student 下降 `99.9716%/99.5570%/99.2316%`。torch export 到 hash-bound
pure-NumPy artifact 和在线逐步 replay 的最大误差分别为 `2.22e-16/1.11e-16`；leakage 或任一 health failure
reset state 并输出 exact zero residual。六个 2/4-state candidates 在 900 epoch cap 达峰，所以不声称 optimizer
全局收敛。以上只证明 matched-model teacher-action imitation；physical lifetime/fidelity/`p(g)`/e-leakage
burden retention、量化、RTL、FPGA/board 继续由 T4.4.4 以后验证。

### 3.41 Paired physical teacher-student gain retention

T4.4.4 在全新 8×64 cutoff12 与 4×32 cutoff16 paired seeds 上冻结重放 standard、全部五个 exact-budget
MF agents、teacher、handcrafted recurrence 和 student；独立 exact 2-cycle lane 才加入 16-branch control oracle。
control oracle 没有被 reset 或外推为 10-cycle reference。student gain retention 定义为
`(student-standard)/(teacher-standard)`，selection score、fidelity lifetime、logical-Z lifetime 的 point 和
20,000-repeat paired-bootstrap CI lower 都必须超过预注册 `0.90`。

cutoff12 三指标 point 为 `0.998481/0.997368/0.996557`，最低 CI lower `0.982442`；cutoff16 为
`0.987812/0.989806/0.981457`，最低 lower `0.944501`。student/teacher 的最大 `p(g)` 差为 `0.002328`，
95 scalars/87 analytic MAC 相对 teacher 降低约 `99.87%/99.88%`。但 cutoff12 五-agent MF mean 略高于
teacher，cutoff16 才反转；handcrafted 的高 `p(g)` 伴随约四倍 residual RMS；multilevel leakage 为 `null`。
因此只支持 student 保留 teacher-vs-standard matched-model gain，不支持 universal NMF-over-MF、leakage/OOD/
long-horizon、量化、RTL、FPGA/board 或 device claim。

### 3.42 Fail-closed teacher-student branch freeze

T4.4.5 不再运行或调参 evaluation，而是逐项复核 T4.4.1--T4.4.4 的 72 个 machine gates、4 个
implementation hashes、7 个 checkpoint/student/source-file bindings 和预注册 retention 语义。当前 8/8
evidence predicates 通过，激活 `qualified_student_retention`；112-row ledger 同时冻结 active/prohibited claims
和后续 revocation triggers。

qualified branch 只允许 matched two-level finite-cutoff、dual-cutoff、新 paired seeds 下的 4-state student
retention 和 float 解析压缩。cutoff12 的 MF mean/teacher selection score 为 `0.557115/0.552952`，cutoff16
为 `0.579684/0.593930`，所以跨 cutoff reversal 必须保留，universal NMF-over-MF 仍被否证。任一 parent
evidence 或 T5.2/T5.4/T5.5/T6 后续 gate 失败，将自动激活 `drift_regime_aware_map_lut` 并删除 teacher/
distillation active claims；leakage、OOD、long-horizon、RTL、FPGA/board 与 device 结论继续关闭。

### 3.43 文献趋势 reproduction registry

T5.0.1 将 2020/2023 实验、NMF PRL、Knill/qunaught、P-Steane、noise-transfer 和 2026 trapped-ion
证据拆成 14 个 machine-readable target。每行都固定 hierarchy role、数值或方向目标、容差、
`calibration_only`/`independent_holdout`/`future_holdout_preregistered`/reference/reporting 用途、当前状态、
后续 gate 与禁止迁移。当前 artifact 的 17/17 registry gates 通过；这只表示表格和 provenance 完整，
不表示 5 个 `REGISTERED_PENDING` 行已复现。

现有 T2.0.5/T2.0.6/T2.3.3/T2.3.7 结果只按其原有模型边界登记为 qualified directional evidence；
T4.4.5 的 exact-budget MF 跨 cutoff 排名反转仍是强制 counterevidence。Knill/qunaught、P-Steane 和双模
trapped-ion 全部保持 secondary；外部微秒、gain 和 lifetime 只作 reference/report schema，不参与本项目
选模或 pass/fail。T5.0.2 才执行独立 holdout。

### 3.44 独立 cross-fidelity holdout

T5.0.2 在代码中预冻结 `2.5/10.25/11.75 dB` 和四个 fresh seeds，排除 T2.3.3 的
`3/5/8/10/12 dB` calibration grid 以及 API reconnaissance 使用过的 `4/11/14 dB`。主 family 没有
重选：两个高 squeezing 点的 noise-transfer/direct-syndrome、Fock/direct-syndrome、canonical q/p、localized
与 clipping gates 均通过，但 `10.25 dB` pooled effective-vs-noise z-score 为 `2.293338`，超过冻结 `2.0`
门；`11.75 dB` 为 `1.867765`。因此主 family 明确为 `FAIL`，不能升级 T5.0.1 的 main validity claim。

同一任务的 secondary P-Steane family 使用论文 Eqs. 36/37 的独立 coefficient covariance propagation 对照
Eqs. 40/41，并在 3 个 ancilla noise、3 个全新 data/ancilla variance ratios、7 个 `b`、4 个整数 `m=2a/b`
上形成 252 点解析 holdout。最大 covariance/product 相对误差为 `5.55e-17/4.80e-16`；所有 `k>1` 网格中
`m=1` 即 `2a=b` 为唯一 argmin，ME-Steane 与 teleportation special cases 也通过。该 family 为 `PASS`，
但仍是 secondary small-noise analytic evidence，不进入 sBs 主排名，不代表 FPGA 实现 physical squeezing。

### 3.45 Lane-aware complete comparison set

T5.1.1 冻结 19 个 comparator 和 8 个公平 lanes，不建立全局排行榜。current-syndrome、continuous-drift、
episode-memory、finite-energy-effective、protocol-wallclock、matched-control、event/regime component 和
short-horizon control-oracle 各自保存 decision target、information set、metric/time/compute contract 与
deployability。只有 decoder oracle 可在 evaluator 读取 hidden DriftState，且保持 nondeployable；control
oracle 只限 exact two-cycle tree，不能外推 10-cycle。

此前缺失的 no-correction anchor 已实现为同 finite-cutoff 初态上的 pure idle-loss channel：10 us 网格上
gate/measurement/reset/frame/update 计数全为 0，且 10 us 与 2×5 us semigroup 最大差 `1.11e-16`，不是
standard sBs 曲线改名。finite-energy static 也实际运行 120k train/300k held-out 的五点 shrinkage sweep，
而不是 contract-only 标签。16 个 parent artifacts、19 个代码锚、100-row Source Data 和 14 gates 均通过。
T5.1.1 冻结时该 artifact 的 matrix 状态为 `PREREGISTERED_NOT_EXECUTED_T5_1_2`；随后 T5.1.2 已按 3.46
另建并执行 lane-local matrix，没有回写或伪造 T5.1.1 历史状态。run-length/HMM 未接 logical-policy adapter
前只作 component，Knill/P-Steane 不进入 sBs 主排名，MF cutoff reversal 与禁止 universal NMF claim 保留。

### 3.46 Lane-local mixed scenario matrix

T5.1.2 已执行 static Gaussian、mean/variance/correlation drift、loss、readout/ancilla drift、burst/outlier、
large-error recovery、leakage 和 calibration shift 共 10 类场景。六个 decoder 场景共 36 个 seed-cluster、
589,824 个 paired decisions；standard/static/latest-window/EWMA/Kalman/oracle 在同一 scenario-seed-window
消费同一 trace，predictor 只在当前窗解码完成后更新，训练与评估 seeds 严格分离。

loss 使用原生 noise-transfer bias/covariance/alias 指标，readout/ancilla 使用 protocol-native fault overlay，
large-error 与 leakage 保持 `component-only` native gates；四类结果不得与 syndrome LER 混成一个分数。
15/15 gates 只证明覆盖、因果、物理/统计自检和 provenance，不代表任一 comparator 全局胜出。相关轴下
joint jump 只允许 Fréchet bounds；本次 loss isolation 使用 diagonal axes 才报告 exact joint probability。
theory-only Steane、Knill/P-Steane 均保持不可执行且不进入 sBs 主排名。正式 average/tail、双 oracle-gap、
bootstrap 和多重比较在 T5.1.3 单独执行。

### 3.47 Paired-seed tail 与双 oracle-gap

T5.1.3 按 T5.1.2 相同 RNG 顺序重放 36 个 scenario-seed clusters，保存 1,152 个 windows；所有 trace hash
与 seed rates 精确一致。6 个 seeds 是独立 cluster，20,000-repeat bootstrap 重采样整条 seed trajectory，
报告 syndrome-level `P_L`、window p95、observed worst 和 decoder-oracle gap。24 个 challenger-vs-static
hypotheses 用 `2^6` exact sign flips 和 Holm-Bonferroni；当前最小 raw/adjusted p 为 `0.03125/0.75`，正式
discovery 为 0。该结果进入 T5.1.4 证否门，不得把正 bootstrap effect CI 单独升级成主张。

calibration shift 中 Kalman average/p95 低于 static，但 worst 为 `55/512`，高于 static 的 `37/512`；该
causal transient 必须保留。control lane 只消费 cutoff12/16、exact two-cycle、16-branch expectation，不造
sampling CI。注册的 control oracle 是 finite-multistart matched-model reference，不是全局定理上界：不同
metric 与 frozen cutoff transfer 可产生负 gap，且不能外推 10 cycles。decoder/control metrics 始终分 lane。

### 3.48 Fail-closed 算法成功/证否分支

T5.1.4 只读绑定 T5.1.1--T5.1.3、T2.4.3、T4.2.1、T4.2.3、T4.3.2 与 T4.4.5，
不新增 evaluation 或重选场景/方法/指标。强分支要求 matched learned decoder、同轨 strong baseline、static
average/p95/worst 不退化、独立 seed-cluster advantage、Holm-adjusted discovery、无 transient tail violation
及 observed-only/deployment scope 全部成立。

当前 decoder lane 没有 learned candidate，24-test Holm family 为 0 discovery，且 calibration-shift Kalman
worst `55/512` 高于 static `37/512`。因此唯一激活分支为
`event_aware_adaptive_map_fpga_codesign`；CNN/TCN/GRU performance claim 被删除。该名称表示后续论文与
工程方向，不能被解读为 run-length/HMM 已接入统一 finite-energy closed loop 或 FPGA/board 已测。

T4.4.5 qualified student-retention 仍只属于 matched two-level controller lane，历史 T24 只属于 frozen-set
scope；两者均不能升级为 T5.1 decoder confirmation。未来重开 strong branch 必须在访问新数据前预注册
independent seed clusters，并同时通过 same-trace、static/tail、Holm、causality 和 deployment gates；现有
1,152 windows 禁止改名为独立 seeds。

### 3.49 物理时间与控制成本公平 lanes

T5.1.5 将时间与成本证据固定为三条不可混排 lanes。protocol lane 在 cutoff12/16×三噪声下使用共同
700 μs：measurement-feedback 为 10 μs/cycle、70 cycles、140 measurements/140 resets/1260 active gates；
autonomous 为 7 μs/cycle、100 cycles、0 measurements/200 resets/1800 active gates。6/6 场景按 protocol
cycle 的 autonomous ratio `>1`，按共同 μs 的 ratio `<1`；两种口径必须同报。

matched controller lane 在 cutoff12/16 下统一 10 cycles/100 μs，并为 standard、exact-budget MF、teacher、
handcrafted recurrence、student 同报 cycles/μs、20 measurements、20 resets、180 active gates、`e` outcome
burden、scalars/MAC 和 latency availability。`e` events 不是 reset；two-cycle finite-horizon reference 不进入
ten-cycle table。

host estimator lane 只保留 T4.1.1 development-host batch medians，physical lifetime/events 为 null，不能转移
给 T4.4.4 controller 或 FPGA。项目配置的 `1.0/995 μs` 是 assumption；target-board core/transport/end-to-end
与 measurement/ADC/AWG/physical-action latency 共 7 项仍为 null。不得构造跨 lane aggregate latency 或总分。

### 3.50 实验可行性约束与非混合门

T5.1.6 只读绑定 T4.4.4、T5.1.5、T4.3.3、T4.2.3、T4.4.3 与 T5.1.4。matched-controller
lane 的 `p(g)/p(e)` 只属于 two-level simulator；每行 20 resets 与 e outcomes 分列，multilevel leakage、
parameter saturation rate 和 matched latency 保持 null。cutoff12/16 峰值 lifetime 均携带完整 scalars/MAC、
reset 与 latency-null，不能支撑 deployment claim。

fault lane 保留 8×4 software campaigns、767,872-cycle denominator、11,552 fallback、4 reset requests 与
0 observed unsafe/undefined actions。该定向 campaign 是 software coverage evidence，不是 device-fault population
抽样，因此不构造总体置信上界。component fallback taxonomy 和 student fail-closed contract 仍是独立证据层。

controller multilevel leakage、saturation、matched/board/frontend latency、device reset fidelity 与同一
finite-energy closed-loop 的 lifetime/fault-rate 联合证据共七项保持 `MISSING`；总体 deployment readiness 为
`NOT_ESTABLISHED`。

### 3.51 Displacement / large-distance 独立因果 lane

T5.2.1 不复用 T2.0.5 的单 seed 结果作为正式结论，而是在冻结 recovery/readout/reset kernel 后执行
17 幅度×8 evaluation seed clusters×4,096 shots。每个 seed 内用 common random numbers，只改变 nominal
displacement amplitude；CI 按 whole-seed cluster bootstrap，不把 shots 当独立实验。initial depth 与 observed
same-quadrature e-run 均在 `epsilon/l_S=0.25` 达峰，midpoint mean 为 `6.00000/4.84512`。

logical assay 再分成两个不可互换的 estimand：nearest-operation-relative misclassification 在 midpoint 约 0.5、
两端接近 0；identity-reference flip 从 0 单调升到 1，明确保留 `epsilon/l_S=0.5` logical operation。actual
parity 与 nominal target 只供 evaluator；logical failure 不是 recovery censoring，也不是 repeated physical-memory
LER。两个冻结 jitter profiles 与 Gaussian boundary formula 的最大绝对差 `<0.005`。

该 lane 仍使用 coarse error-space 与 project-assumption recovery/readout/jitter，不是 coherent Fock injection、
Fig. 4(c) 定量拟合、device calibration、QPU 或 target-board fault injection。

### 3.52 Ancilla bit/phase 与 readout 独立因果 lanes

T5.2.2 不把 T2.2.2/T5.1.2 的 mixed aggregate 改名为单通道因果证据，而是对 bit-only、phase-only 和
readout-only 分别执行 6 个 rate×8 个新 seed clusters×4,096 cycles。每个 family/seed 内跨 rate 使用 common
random numbers，20,000-repeat CI 重采样整条 seed cluster；balanced `K_gg/K_ge/K_eg/K_ee` schedule 同时覆盖
g/e 与 X/Z。每行保存 11 个主/交叉 estimands，所有未注入通道必须精确为 0。

bit-only 只改变 X big-CD bit probability，event/toggle/logical backaction 随 rate 增长；phase-only 只改变同一
位置 phase probability，连续小回作用增长，但 Z-basis toggle、logical backaction 和 ideal-label change 精确为 0；
readout-only 只改变 hidden g/e classifier confusion，misclassification 与 virtual rotation 增长，但不产生 ancilla
fault event 或 logical truth。readout 误判导致错误 registered reset action 是其下游因果结果，不是额外注入
reset-failure channel；后者留给 T5.2.3。

三族不合并 global sensitivity score。0.5 bit-to-logical conditional、0.01 phase scale、symmetric confusion 和
0.6 rad range 均为 project assumptions；Sivak S4I 的 65× 只作定性机制锚，不是数值复现目标。truth 只供
evaluator，产物不是 cavity--transmon master equation、physical-memory LER、device calibration、QPU 或板测。

### 3.53 Leakage 与 reset-failure 独立因果 lanes

T5.2.3 不把 T2.0.6 的 occupancy/post-selection component 或 T5.1.2 aggregate lane 改名为在线闭环证据，
而是分开执行 leakage-injection-only 与 reset-failure-only 两族，每族 6 个 rate×8 个独立 seed clusters×
256 trajectories×512 evaluation cycles；另有 128-cycle burn-in。每个 family/seed 内跨 rate 使用 common
random numbers，20,000-repeat CI 只重采样整条 seed cluster。leakage lane 固定 reset failure，reset lane
固定 leakage injection，并用 empirical intervention/fixed-channel rate gate 防止双通道静默变化。

检测器只消费 observed `g/e/leakage`，hidden leakage truth 只供 evaluator。无真实 leakage episode 时，
detection fraction 与 delay 保持 `null`，而 false-alarm rate、declared/safe availability 仍可定义；不以伪造的
零延迟填补空集合。raw reset request、attempt、success、failure、occupancy/run length 以及 lag 1/2/4/8/16/32
correlation/covariance 分列，任何 intervention 均不通过 post-selection 丢弃 trajectory。

0.95 detector sensitivity、`2e-4` false-leak alarm 和两条 intervention law 都是 project assumptions。有限
formal sample 中 detection fraction 为 1 不构成总体保证，safe availability 也不是 board uptime。产物不是
multilevel master equation、physical-memory LER、device calibration、QPU 或 hardware fault injection。

### 3.54 六态 CPTNI logical-channel reconstruction

T5.3.1 不把 T1.2.2 parity-twirl、T4.4 learned trajectories 或 T5.2 component fault estimands 拼成
logical channel。唯一 channel lane 复用 T3.2.8 的 finite-cutoff nominal sBs map；QEC-on/off 使用同一
orthonormal code isometry、cavity/ancilla noise、初态、cutoff 与 `10 us` reporting interval，只有
`qec_on` nominal sBs 对 `qec_off` matched idle 的 intervention 不同。

每条 lane 输入 `X+/X-/Y+/Y-/Z+/Z-`，逐 cycle 保存 unnormalized `2x2` code outputs，并重构线性
CPTNI PTM、Choi/TNI、non-Pauli、non-unital、state-dependent survival 与 missing-trace leakage。
conditional Bloch 只作诊断，不进入 PTM；lifetime 使用 raw code-weighted Pauli contrast 的 finite-horizon
signed area 与真实 e-fold crossing/censoring，不做单指数拟合或 postselection。

formal matrix 为 4 cutoffs×3 noise profiles×on/off×30 cycles。低 cutoff 12 的性能方向与高 cutoff 相反，
因此保留完整 12/24/36/40 scan，并要求 terminal 36→40 的 PTM/leakage 数值稳定；该 repeat 不是
infinite-cutoff theorem。所有 raw 六态 outputs 必须能重算 survival、PTM/Choi、lifetime、matched comparison
和 cutoff tables。结果不是 experimental tomography、multilevel leakage、physical-memory LER、break-even、
device calibration、QPU 或 FPGA timing。

### 3.55 CPTNI fidelity 与 short-time rate

T5.3.2 只消费 T5.3.1 同一组 24 条六态 raw channel lanes，不从 parity-twirl、conditional Bloch 或历史
lifetime 表拼装 fidelity。对 trace-nonincreasing code subchannel 使用 `F_e=Tr(R)/4` 与
`F_avg=(2F_e+R_II)/3`；常见 TP 公式的高估量 `(1-R_II)/3`、六态 direct overlap、mean survival 和
conditional normalized diagnostic 必须分列。conditional 指标不作为线性 channel fidelity。

短时率定义为 `Gamma=-2 dF_avg/dt|0`，primary 是 10 us grid 的三点二阶 forward difference；一/四点
只构成离散化敏感度，不做单指数拟合。只有前 3 点非增且多阶 rate 相对 spread 不超过 25% 才报告
qualified inverse-rate lifetime。正式 qec-on lanes 因 cycle-scale recovery transient 全部不合格，raw rate
保留而 lifetime 为 null；qec-off qualified proxy 不得跨口径升级为 active break-even。

formal exact propagation 没有 sampling clusters，因此 standard error/CI 保持 null。36/40 cutoff interval 与
1/3/4-point envelope 都是 deterministic systematic sensitivity，不是统计 CI 或 infinite-cutoff theorem。
cutoff12 的 on/off 方向反转必须保留；matched differences 不转成 ratio/gain。结果不是 experimental
tomography、physical-memory LER、simulated break-even、device calibration、QPU 或 FPGA 结果。

### 3.56 full-curve wall-clock operational boundary

T5.3.3 同时绑定 T5.3.1 channel 与 T5.3.2 fidelity artifacts，只比较相同 code basis、cutoff、noise、初态、
10 us grid 和 300 us horizon 下的 fixed nominal sBs 与 matched idle。qec-off 的角色固定为
`matched_uncorrected_grid_code`，不是论文中的 best passive physical qubit。

primary sampled boundary 是最后一个 `F_on-F_off<0` 样本之后的首个样本，且 active 必须在所有后续样本
持续非劣；cumulative payback boundary 还要求 `integral(F_on-F_off)dt` 偿还早期 deficit 并持续非负。
两者都消费完整 31 点 leakage-inclusive `F_avg` curve，不以单终点、area ratio、单指数或不合格短时率替代。
线性零点只作相邻采样诊断，不提供 sub-grid validation。

formal scan 保留 12/24/36/40×3 noise。cutoff12 无边界是必要反例；36/40 terminal repeat 才允许写
`simulation-derived wall-clock operational boundary`。由于 active rate 不合格、best-passive reference 缺失且
active pulse/reset/classical/latency 尚未统一计价，paper-defined coherence gain、full-cost boundary 与
experimental/physical-memory break-even 均保持 `NOT_ESTABLISHED`。

### 3.57 online QEC 与 post-selection 成本隔离

T5.3.4 直接读取 T5.3.1 qec-on native `event_accounting`：300 us/30 full cycles 含 60 measurements、
60 resets、540 active gate applications；qec-off 三项为 0。T5.1.5 standard protocol 只作独立计数复核，
不把 cutoff12/16 controller performance 转移到 cutoff36/40 channel。`Delta=0.34` 按项目约定报告
`-10 log10(2 Delta^2)=6.360122 dB`；15 scalars/0 online policy MAC 是 fixed nominal analytic resource，
不是 RTL/board/pulse-energy 证据。

online channel 无 post-selection，trajectory acceptance 固定 1；CPTNI code-space survival 仍小于 1，两者
不得混名。online achieved metric 只用 parent `F_avg`/survival/boundary，physical-memory LER 保持 null。

T3.2.4 的 post-selection 独立报告 acceptance、rejection、conditional decision error、accepted failures 和
`accepted failures + lambda*rejection` 四档成本。它属于 synthetic decoder diagnostic，不填 `F_avg`、LER、
event count 或 latency，不与 online channel 拼 total。truth upper 只供 evaluator；conditional improvement 不
进入在线主增益。T5.1.6 safety burden 也保留为另一条 deterministic software campaign。

matched/board/frontend latency、pulse duration/energy、device reset、best-passive reference 和 matched LER 等
12 项缺失字段保持 null。没有跨 lane global score；full-cost operational boundary、paper coherence gain 与
postselected/experimental break-even 均保持 `NOT_ESTABLISHED`。

### 3.58 QEC-matrix/Petz arbitrary-recovery bound

T5.3.5 的对象是 isometric finite-cutoff GKP encoding 经过单个 `10 us` exact cavity pure-loss channel 后，
允许任意 CPTP terminal recovery 所能达到的 channel/entanglement fidelity；它不是 per-shot decoder、
history controller、sBs pulse sequence 或 FPGA runtime。QEC matrix 使用
`M[mu,l;nu,k]=<mu|A_l^dagger A_k|nu>`，Petz fidelity 为
`F_tilde=||Tr_L sqrt(M)||_F^2/d_L^2`，解析区间固定为
`F_tilde <= F_opt <= (1+F_tilde)/2`。

small-cutoff formal matrix 为 cutoff 4/6/8/10/12×high/medium/low。每条 lane 独立求 recovery-Choi
primal 与 dual SDP；raw solver 状态、残差和 duality gap 原样保留，再把 primal 投影为 PSD 并以 partial-trace
inverse square root 归一成 CPTP 可行点，把 dual 乘积 slack 的负最小特征值用 identity shift 修复。因此只在
repaired primal lower 与 shifted dual upper 相交时通过，不用启发式搜索冒充 optimal recovery。

无需 SDP 的 Petz/QEC-matrix scan 扩展到 cutoff 12/24/36/40/48，并在 cutoff 24/36/48 对
`Delta=0.44/0.34/0.28` 做能量敏感度。高 cutoff 的 noise-output support inverse square root 会病态，
因此同时保存 QEC-vs-direct Petz fidelity residual、support TP residual 与 cutoff difference；cutoff 48 不是
无限维或无限能收敛证明。

actual nominal sBs 在两次 half-cycle 中交错 gate/reset、cavity 与 ancilla noise，不是“先纯损耗、再单次任意
recovery”。其 CPTNI `F_e` 只在补上 maximally-mixed leakage completion 后，与 bound 报一个
`SCHEDULE_MISMATCHED_DIAGNOSTIC_ONLY` 差值，不声称 certified ordering。T4.4.4 teacher/student 只有
10-cycle two-level trajectory lifetime/score，没有 matched six-state one-cycle Choi；二者 gap 必须为
`null/INCOMPARABLE`，禁止 lifetime 减 `F_e`。结果不是 deployable recovery、large-cutoff SDP optimum、
physical-memory LER、device calibration、QPU、RTL 或 hardware result。

### 3.59 Held-out/OOD lane-local contract

T5.4.1 不把 T5.1.2 的 mixed matrix、T5.2.2/T5.2.3 的正式 rate grid 或 T2.4.2 的旧通信场景改名为
OOD。访问结果前冻结 32 个全新且两两互斥的 seed clusters，并与 parent artifacts 中递归提取的
train/validation/evaluation/bootstrap seeds 做集合隔离；T5.1.2 的 static MAP 与 EWMA/Kalman 参数按 parent
hash 恢复，OOD 数据不参与 refit、threshold 或 method selection。

四条原生 lane 分别是：unseen joint-sinusoidal/telegraph family 与 compound range extrapolation 的 paired
syndrome-decoder replay；3 张新 4×3 row-stochastic sBs confusion matrix；`0.003/0.006/0.012` persistent
leakage-rate replay；以及 micro-outage/increasing-flap/communication+jitter+burst scheduler stress。每条 lane
只报告自己的 decision/proper-score、confusion、leakage burden 或 LER/availability/event observable。

telegraph 下 adaptive 点估计反转但 cluster CI 跨 0；periodic micro-outage 被检测但对当前 metric 为 null；
increasing/compound communication faults 显著退化。这些负结果必须保留，不能用 expected-direction gate
删除。20/20 gate 的 `PASS` 只表示 104 个 seed cells、280-row Source Data、provenance、数值和 fail-closed
integrity 完整；`system_robustness_status` 与 `device_robustness_status` 保持 `NOT_ESTABLISHED`。T5.4.2 才能在
matched fault population 上比较 uncertainty-gated fallback 与 no-fallback。

### 3.60 Matched uncertainty-gated fallback contract

T5.4.2 的 no-fallback 是 T5.1.2 frozen EWMA periodic MAP，last-known-good fallback 是同一 parent 的
frozen static MAP；两者消费同一 residual 和同一 hidden evaluator truth。score 只由当前 modular-syndrome
posteriors、past-only one-window-delayed Window/EWMA/Kalman states 构成，在本窗 predictor update 前决定动作；
truth 只作离线计分。catastrophic failure 是 decoded logical class 错误，不是 OOD/health/uncertainty proxy。

41 点 threshold grid 只用 T5.4.1 的 8 个 development seed clusters 选择，再冻结 `0.45` 并运行 12 个
parent-disjoint confirmation clusters。36 个 OOD cells/1,179,648 decisions 上，primary/gated error 为
`0.04718865/0.04611376`，absolute reduction 为 `0.00107490 [0.00001950,0.00227615]`；同时报告
17,788 fallback、6,170 avoided、4,902 induced、7,093 unnecessary 和 11,618 selected-without-benefit。

逐场景结果不合并成普适 claim：telegraph reduction 的 CI 全正，compound reduction
`-0.00488536 [-0.00557454,-0.00418854]` 全负，sinusoidal CI 跨 0；nominal negative control 也有
`-0.00001272` 点估计代价。21 gates 和 517-row Source Data 只证明当前 matched synthetic
syndrome-decision mixture 的聚合效果与成本可追溯；不证明 physical-memory LER、universal OOD safety、
controller/RTL/board 或 device fallback。

### 3.61 Native-lane causal ablation and negative-result contract

T5.4.3 先验证六项 requested switches 是否共存于一个已验证端到端 stack。答案是否定的：history 使用
T3.2.11 finite-cutoff control lifetime，CNN residual 使用 preserved legacy parameter-prediction split，regime
state 使用 T3.2.6 proper score，run-length/parameter update 使用 T3.2.5 software event cost，fallback 使用
T5.4.2 syndrome logical-class failure。因此每个 mechanism-off 只在自己的 native matched lane 内成立，
`cross_lane_aggregate` 和 `global_ranking` 必须为 null。

history 的 full-minus-latest-only lifetime 在 cutoff12 为
`0.709110 [0.568837,0.894025]`，cutoff16 为 `-0.563636 [-0.665556,-0.461717]`；run-length
相对 memoryless 的 cost benefit 为 `-0.179911 [-0.180782,-0.179041]`。两项负结果不得删除。
CNN residual 的 active/off MSE 为 `2.41445e-6/8.03405e-6`，但只有一个 legacy test split、无独立
seed-cluster CI，T5.1.4 撤销决定不变。regime NLL benefit 与 detection-delay cost 分别为
`0.401514 [0.366352,0.436676]` 和 `1.228754 [1.111948,1.345561]`；parameter update 只允许作为
event-actuation component result；fallback 保留 aggregate 正向及 compound/nominal 负向。

18 gates 与 338-row Source Data 只证明六项原生 intervention、provenance、符号、代价和 claim 降级完整；
不证明六机制联合归因、普遍收益、physical-memory LER、device calibration、RTL/FPGA/board 或实验结果。

### 3.62 Validation-only all-agent/seed selection audit contract

T5.4.4 只读重构 T2.3.7、T4.1.1、T4.1.5、T4.4.1、T4.4.3 与 T4.4.4 的 selection。
candidate 只允许由 validation 决定；独立 test 只能评估或计算不参与重选的 hindsight optimism。对越大
越好的 metric，worst quartile 取最低 `ceil(n/4)`；对越小越好的 metric 取最高 `ceil(n/4)`；全部
distribution 同报线性 Q1/median/Q3/IQR。

NMF-minus-MF logical-Z lifetime 的五 agent primary median/IQR 为 `0.257219/0.242015`，confirmation
为 `0.386384/0.175685`；若违规只选 test-best agent，相对 median 会分别夸大 `0.133006/0.394202`
cycles。T4.4.4 仍保留全部五个 MF agents。fresh teacher validation 选择 restart 0，但 primary/confirmation
test-best 均为 restart 2，潜在 optimism `0.004127/0.001445`；candidate 不因此改变。slow-loop HMM 与
4-state student 的 validation/test 排名一致。

旧 T4.1.5 只保存 selected restart 的 evaluation，nonselected test counterfactual 不可复算；该 predecessor
标为 superseded warning，不伪造完整性。23 gates、255 evaluation units、39 distributions 与 420-row
Source Data 支持 `PASS_WITH_WARNINGS`：证明 selection discipline 和全体报告完整，不证明无 selection-bias
风险、optimizer optimality、universal memory gain、physical-memory、device 或 hardware 结果。

### 3.63 Training-horizon and real long-recurrence contract

T5.4.5 用同一 frozen GRU teacher 与 split-specific 64-half-cycle histories，把 2/5/10-cycle students 各重新
拟合三个 restarts；只由 validation 选择。32-cycle production student 直接绑定 T4.4.3，不用本任务 evaluation
重选。2-cycle candidate 在短 validation 很低，却在独立 32-cycle evaluation 达 `9.9540e-5`，其 all-e long
worst 达 `8.1423e-4`；该 horizon failure 必须保留。

部署矩阵为 stationary/persistent/range-shift 各两 seeds 加 all-g/all-e，共 8 条 streams；每条真实消费
2,000,000 half-cycle tokens。GRU-10 与四个 student models 在 float64/float32 下均全步递推；dense action
只在 13,631 个冻结 checkpoints 计算。teacher hidden 最大 `0.382517<1`；student actual state 不超过初态与
两 outcome saturation 的解析凸包；float32 最大动作差 `1.2630e-6`。终点前 256-half-cycle 强制 state reset
形成 120 个 matched counterfactual，最慢恢复 20 half-cycles。

10/32-cycle models 在 `1e3/1e5/1e6` cycles 的 sampled teacher-action mean 与 worst-stream MSE 均低于
`5e-5`，但这只是 observed `g/e` recurrence/action imitation。没有执行百万-cycle Fock logical channel；
T4.4.4 的 10-cycle physical retention 不得由 action MSE 外推。physical-memory LER、long-horizon physical
gain、leakage/model-mismatch、device、RTL/FPGA/board 与实验 claim 继续关闭。

### 3.64 Randomized multi-factor mismatch contract

T5.4.6 将异构 mismatch 保持在四条不可拼榜的 native lanes。finite-cutoff physical control 使用 cutoff12、
10 cycles、batch16、float64；32 个 cells 分为 full 15-vector gate bias、cavity phase diffusion、固定
`5000 ns` 总时长的 phase allocation/lifetime dynamics 与 compound 四族。每个 cell 对
standard/teacher/student 执行 nominal/mismatch 配对；branch 必须从 raw strategy scores 重新计算，不能读取
stored PASS。另有 8 个完整 4×3 readout matrices、16 个 persistent leakage/reset cells 与 8 个
parent-frozen random-drift cells；readout 的 f/higher 行通过独立 categorical calibration 实际访问，但不冒充
multilevel master-equation trajectory。

全部 64 个 vectors 与 1,351 个 parent seeds 无重叠；19 gates 和 273-row Source Data 通过。32/32 teacher
gain qualifying；student retention median/Q1/min 为 `0.998101/0.990413/0.897630`，compound median
`0.995598`，所以 qualified student branch 保留。该 relative retention 不掩盖 absolute degradation：
gate-bias/compound teacher worst nominal-minus-mismatch score degradation 为 `0.424155/0.395654`。

随机分布不是 device posterior；physical lane 只是 10-cycle finite-cutoff evidence；readout categorical、
effective leakage kernel 与 syndrome drift 不形成 integrated score。long-horizon physical channel/LER、
device calibration、multilevel leakage/SPAM、RTL/FPGA/board 和实验 claim 继续关闭。

### 3.65 Packed-word bit-accurate RTL-golden contract

T5.5.1 把 T4.2/T4.3 的整数组件组合成真实逐周期 Python golden，而不是继续用预构造的
`input_cycle=cycle-5` decision。online input/output/state 分别为 58/118/232-bit CRC words；每周期先发布
上周期 output，再 atomic commit、S0 锁存 input+image+version、推进五级 MAP/FSM，并把 action 注册到下一
周期。因此 source-to-output 恰为 6 cycles、II=1。4,110 个连续 inputs 的 source 0--4109 均唯一输出，
没有 warm-up 后丢失或重复。

parameter image 使用 128-byte header、两个 257×signed-24-bit containers（logical Q9.12 为 signed
22 bit）与 CRC32+SHA256 trailer；8 images/bundle 全 exact roundtrip。cycle 4000 unsafe defer、4001
commit v1；source4000 保持锁存 v0，source4001 锁存 v1。16,384 个 code 的 LLR/action 与独立整数
ties-to-even 重构完全一致；4,116-row trace 有 output/state CRC 和逐行 SHA256 chain。

该 contract 从 ADC code 开始；raw IQ/ADC、CDC、transport 与 physical action 不在内。24-bit container 是
表示选择，不是 BRAM packing 结果。RTL、synthesis/post-route、Fmax、LUT/FF/BRAM/DSP、board 和 device
evidence 仍为 null/false，必须由 T5.5.2/T6 独立升级。

### 3.66 Synthesizable RTL and target-device post-route contract

T-RISK-20260716-01 把 T5.5.1 packed words 映射为可综合 RTL。core 保留 58/118/232-bit CRC words、
exact ties-to-even interpolation、5+1-cycle pipeline、双 bank atomic switch、event/fallback/frame/counter
state。四个逻辑 257×22-bit 2R1W tables 用八个 mirrored 1R1W BSRAM 实现；动态写入广播到读镜像。
CXXRTL fault/commit 与 exhaustive v0/v1 traces 共 4,316 个 valid MAP rows，map/output/state/version/ack
逐周期 0 mismatch。该等价性不含 raw IQ/ADC、CDC、transport 或 physical action。

T5.5.2 在固定 YoWASP Yosys/nextpnr/Apycula 上，以 Tang Nano 20K 的
`GW2AR-LV18QN88C8/I7`、`GW2A-18C`、27 MHz SDC 和 QN88 small-pin CST 运行真实 synthesis/P&R。
复核把 harness 配置地址严格约简到 0--256 后，seed 1/7/19 Fmax 为
`40.4318/39.8661/39.7456 MHz`；报告必须使用最差 seed，不能挑选最佳值。三 seed最大
LUT4/DFF/BSRAM 为 `3362/865/8`，MULT18X18/MULT9X9 各 1；最差 critical period
`25.1600 ns`，从 core state register 到 activity harness `fold5` 观测寄存器。core latency 为 6 cycles，
在 27 MHz 是 `222.222 ns`，II=1 是 `37.037 ns`。

activity harness 只保持全部输出和配置路径可观测，不是 T6 transport。当前 evidence 可升级为
target-device open-source post-route estimate；vendor timing signoff、bitstream、board、transport、板上
latency/throughput/power 和 quantum hardware measurement 仍必须为 false。

### 3.67 Precision-resource-performance deployment-point contract

T5.5.3 不把 T4.2.4 precision proxy、T3.1.5 top-K operation count 与 T4.4.3 float student cost 直接
相加。它先保留 4×3×3×3=108 个联合 candidates，再要求父 quality gate 与实际 integrated P&R 同时
存在。precision 门选满足 action disagreement `<=1e-4` 的最小 p10/a8/Q9.12；top-K 门选六场景最小
收敛 K=4，但后者仅为 off-device reference，`online_topk_rtl=false`；student dimension 只保留父任务
evaluation-blind 选择的 4-state。

4-state student 用 signed Q3.14、one-multiplier serial recurrence/head；CXXRTL 对 512 operations、7,680
outputs、完整 72-bit state 与 5 forced resets 逐码一致，fixed-vs-float 最大输出差 `1.46038e-4`。集成
共享 harness 地址修复后重新综合/P&R 的 core+student seed 1/7/19 Fmax 为
`40.5351/40.3226/39.5726 MHz`；最大 LUT4/DFF/BSRAM 为 `3802/1022/8`，
MULT18X18/MULT9X9 为 `2/1`。student 64 cycles=`2.37037 us`@27MHz，低于
5 us project-model slot；core 仍并行保持 6 cycles。

只有 p10/K4-reference/state4/P1 带 actual three-seed post-route evidence；P2/P4 和其余组合是明确的
estimate。terminal `bias_mem[15]` 组合越界已由 CXXRTL 暴露并修复为地址钳位。该 contract 不建立
online top-K、full/quantized GRU、vendor signoff、bitstream、transport、board 或量子硬件证据。

### 3.68 Full/quantized-GRU versus distilled-student feasibility contract

T5.5.4 对同一个 selected `GRU10-DENSE256-DENSE256-OUT15` checkpoint 重算 72,853 parameters、
72,266 weight MACs 和 587 biases。float32/float64 parameter storage alone 分别需 127/253 个 BSRAM，
加 8-block core 后是 135/261，均超过 target 的 46；因此完整 GRU 保持 offline teacher，不用删层或
空壳 top 伪造 target synthesis。

量化分支保存全部 588,694 parameter bits。functional fake-quantized shadow 先以 manual gate equation
对 PyTorch `GRUCell` 达到 `5.55e-17`，再覆盖全部 256 个 length-8 histories/all-prefixes 与
128×256 long random sequences；最大 action error `5.5199e-4`，但这不是 physical gain retention。
被综合的 workload 只用于乐观下界：它真实消费 72,266 weights/587 biases，却明确省略 gate dependency、
activation buffer 和 nonlinearities。独立 bit-vector signature 复核发现并修复 bias0 重复、terminal bias
漏读与地址 587 越界；修复后 CXXRTL/reference signature 均为 `730990968`。重新执行的三 seed P&R
共享 harness 修复后再次重跑的 Fmax 为 `40.2625/39.1527/40.6835 MHz`，最大
LUT4/DFF/BSRAM 为 `3904/1011/41`。即使用 min Fmax，下界仍 `1860.76 us`，远超 5 us；当前
inputs/netlist/route artifacts 另由 provenance 绑定。

因此 optional quantized-GRU enhanced route 为 Dropped，唯一通过 functional、capacity、deadline、matched
physical gain 四门的是 4-state Q3.14 student。lower-bound P&R 不能被改写为 functional GRU 或 worst-case
latency；vendor signoff、bitstream、transport、board、power/throughput 与 quantum hardware 仍关闭。

## 4. 交叉验证协议边界

2020 sharpen–trim 以 `+y/-y` 读出控制 feedback shift，按两个 sharpen 加两个 trim 循环。T2.2.2 已实现其协议原生 ancilla/readout/reset effective flow，可用于 fault-information-flow 交叉验证；它仍不能与 sBs 共享 cycle、syndrome alphabet、timing 或主排名。原文给出 conditional-displacement/readout 等组件叙述，却没有足以冻结完整一轮/四轮 wall-clock 的单一数值，所以这里保持 `null`，不做组件加和推断。

## 5. Secondary 范围

- Knill/qunaught：只保留“Knill/Steane 对照、qunaught+beamsplitter 和 secondary 报告的约 `10^-8` 数值等价”作为待核验线索。没有一手全文锚定和独立复现前，不写成项目结果。
- ME/P-Steane：只保留 tunable preprocessing `(a,b)`、noise-ratio scheduling 和 secondary 报告条件 `2a=b`。FPGA 可选择编码后的参数索引，不代表 FPGA 实现物理 squeezing。

两类 secondary 都不得进入 sBs 主排名、不得支撑 cavity/transmon 或 FPGA 物理 claim。

## 6. 禁止 silent protocol mixing

1. 不把 sBs `4.924 us`/`9.848 us` 迁移到 sharpen–trim 或 secondary。
2. 不把 `+y/-y`、单步 `g/e`、成对 `gg/ge/eg/ee` 合并成一个 syndrome vocabulary。
3. 不把 `f` 写成第五个理想 sBs Kraus operator。
4. 不把 cross-validation 或 secondary reproduction 放进主协议排名。
5. 不把 contract、文献事实、软件仿真、综合估计、实板测量合并为“已实现”。
6. autonomous sBs 保留到 T3.2.8 作为 protocol-native baseline，不静默替换 measurement-feedback 主数字孪生。

## 7. 证据锚与后续实现门

一手锚来自本地 Sivak 2023、Campagne-Ibarcq 2020 与 Puviani 2024 正文/补充材料；secondary 只来自 `docs/任务版改进记录/6篇实用论文.md` 与 Zotero 阅读卡。机器 JSON 保存逐条行锚和 expected fragment，测试会直接回读源文件。

主协议已通过 T2.0.2 grouped CPTP/error hierarchy、T2.0.3 hidden/observed/reset、T2.0.4 Table S3 timeline、T2.0.5 displacement trend、T2.0.6 occupancy/correlation、T2.1.1 stream、T2.1.2 memory、T2.1.3 million-cycle/rare-stratum、T2.2.1 finite-squeezing、T2.2.2 protocol-native ancilla fault、T2.2.3 control-imperfection、T2.3.1 generic finite-Fock、T2.3.2 completed analytic SBS one-round、T2.3.8 noise-transfer surrogate、T2.3.3 axis-resolved cross-fidelity、T-RISK-20260714-01 quadrature contract、T2.3.4 differentiable trajectory、T2.3.5 Feedback-GRAPE gradient、T2.3.6 resource envelope、T2.3.7 strict-split directional ranking，以及 T4.4.1--T4.4.5 的 fresh teacher、hidden/control、student、paired physical retention 和 fail-closed branch freeze。T5.4.5 已建立 observed-stream real long recurrence、state bound、float32 与 reset 证据；T5.4.6 已建立四条 native randomized mismatch lanes 与 fail-closed relative student retention。百万-cycle physical channel/gain 与 T5.5/T6 hardware gates 仍关闭；真实 board timing 继续走 T6 evidence gate；secondary 若要升级，还需一手全文、协议原生测试和任务板明确批准。

## 8. 非 demo 审计结论

T2.0.1 没有用占位类或 demo simulator 冒充协议实现；后续只升级直接验证过的 layers。fast MC 通过 1e6-cycle execution、200k analytic calibration、multi-round recovery ablation、cluster CI、known-mixture weighting 和 negative paths；finite squeezing 通过 component covariance、independent LER、ablation 与六点 ideal-limit；ancilla fault 通过 stage cases、schema negatives、persistence 和 2×80k production；control imperfection 通过 code-level quantizers、两种 noncommuting order、2×80k exact moments、bit/latency sweeps、Q4.20 integration 和 100k production；generic finite-Fock reference 通过四点 cutoff、grid/coordinate convergence、Kraus-vs-Lindblad、POVM backaction 与组合 PSD；analytic SBS one-round 又通过独立公式重构、raw/completed 双轨、六逻辑态、photon-error 回泵、100k branch MC 与五点 cutoff；noise-transfer surrogate 通过 independent cell quadrature、解析 propagation endpoint、40万单轴/20万二维 MC、四逻辑态 state/Fock alignment 和低 squeezing 证否；cross-fidelity 再以四个独立 lane、200k/点、六态 protocol metrics、五 cutoff、canonical q/p 与 legacy negative audit 验证；quadrature contract 另有 15 个 machine gates；differentiable trajectory 又以显式 joint gates、SciPy independent reconstruction、21 CPTP groups、四分支 tree、3000-shot sampling、causal history-policy、CPU/CUDA 和 37 direct tests 封住 open-loop/dummy-gradient 简化；Feedback-GRAPE gradient 则用四/十六分支 exact tree、reward/score 分项差分、四步长 sweep、baseline normalization identity、12,288 条 repeated Monte Carlo、CPU/CUDA parity 与 32 direct tests 排除漏 score path、误差相消和单 batch 假通过；resource envelope 再用 65 个隔离点、真实 72,913-parameter causal recurrent policy、reward/score backward、Adam update、CPU RSS、CUDA peak、2--10 cycle full matrix 和两类 frontier 排除单点/forward-only/未触边界假包络；directional ranking 又用 strict train/validation/test/confirmation split、5+5 全 agent、train-only baseline、schema-v3 source-bound checkpoint、512 test trajectories/agent、cutoff-16 confirmation、history-reset 反证保留和 31 项 focused tests 排除 validation leakage、best-agent post-selection、单 seed 与只报标量的 demo；fresh teacher 再用三个新 restart、960 个真实训练 epochs、parent state/seed non-reuse、validation-only selection、双 cutoff held-out、1,074-row Source Data、full-gradient/causality/reload tests 排除旧 checkpoint 换名、单 restart 和隐藏失败；long recurrence 再用四 training horizons、8 条两百万-step streams、双精度 shadow、解析 state bound、最坏流与 120 reset interventions 排除短 smoke、finite-only、短序列重复和均值掩盖；randomized mismatch 再以 64 个 parent-disjoint cells、四条 native lanes、完整 readout hidden-row audit、raw-score branch 重算和 10 类 mutation 排除单 bias vector、stored PASS 与跨 lane 拼榜。cycle-slip detector、长时 physical-channel gain、device-calibrated fault/control rates、multilevel leakage/SPAM、experimental raw data、online leakage control 和 board timing 继续 fail closed。
