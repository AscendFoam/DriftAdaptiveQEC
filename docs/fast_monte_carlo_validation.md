# T2.1.3 高速 Monte Carlo 与 rare-event 分层抽样验证

**日期：** 2026-07-14  
**实现：** `physics/fast_monte_carlo.py`  
**机器产物：** `docs/t2_1_3_fast_monte_carlo.json`  
**证据范围：** vectorized multi-trajectory syndrome-level effective simulator；非装置标定/实板计时

## 1. 为什么不是逐 cycle Python demo

T2.1.1 的对象级 generator 适合逐步审计，但不适合百万周期统计。T2.1.3 采用“少量 round 循环 × 大量独立 trajectory 向量化”：每轮对全部 trajectories 一次性采样/更新，同时保留每条 trajectory 的 residual、recovery depth/axis、persistent leakage 和 logical parity。因此它不是把 1e6 次独立单轮 Bernoulli 冒充多轮 simulator，也没有删除 recovery/leakage memory 来换速度。

100 rounds × 10,000 trajectories 的 production run 共 1,000,000 cycles；本机 conda Python 实测 `0.2531 s`、约 `3.95e6 cycles/s`。这是 host software throughput，不是 FPGA latency/Fmax/real-time claim。

## 2. Vectorized physics/effective path

每轮仍消费对应 `DriftState`：

- correlated q/p Gaussian core 与 shared outlier component；
- `eta=exp(-loss_gamma)` residual attenuation 与 `(1-eta)V_env` loss noise；
- shared half-open lattice binning 和 q→X/p→Z logical parity；
- folded-magnitude recovery-depth injection；
- base/loss/burst leakage hazard、f/higher persistence；
- active leakage depth accumulation、单象限 recovery gain 和跨轮 residual carry；
- 可选 leakage-conditioned logical fault probability（显式 assumption）。

单轮、无 loss/leakage 的 200,000-trajectory test 与解析 periodic Gaussian any-axis error `1-(1-p_q)(1-p_p)` 在 5 SE 内一致。另一个 160,000-cycle causal ablation 证明 active recovery 把 fixed mean-drift logical event rate 降至 no-recovery 的 75% 以下。

## 3. Rare-event estimand 与分层权重

rare mode 定义一个清楚的 trajectory-level mixture：

\[
P_L=(1-p_r)P_L(\text{no extra episode})+p_rP_L(\text{one extra rare episode}).
\]

rare episode 可为 `burst`、`leakage` 或两者，开始轮次均匀，持续时间 geometric；burst 可放大 displacement、增加 mean shift/loss，leakage 可有显式 logical-fault probability。`p_r`、duration、scale 都是 scenario assumption，不是从当前代码反推的装置参数。

simulation allocation 可把大量 trajectories 分给 rare stratum，但最终始终按真实 `(1-p_r,p_r)` 加权。direct tests 用确定性 one-cycle logical burst/leakage 构造证明：

- true rare probability `0.002`、10 rounds 时返回 `P_L=0.0002`；
- raw 50% rare allocation 的未经加权 event fraction 是 `0.05`，不会被误报为 estimand；
- allocation 从 10% 改为 80% 时 point estimate 不变；
- burst 与 leakage 两类均有独立通过案例。

## 4. Confidence interval

primary CI 以 trajectory 为 cluster：在每个 stratum 内重采样完整 trajectory rate，再按 target weight 合成；因此不会把同一 trajectory 的 100 rounds 当 100 个完全独立样本。默认 500 replicates、固定派生 seed，same-seed CI/SE 完全复现。

若某个 stratum 没有任何 failed trajectory，普通 percentile bootstrap 会退化为 `[0,0]`。实现额外使用独立 trajectory 的 exact zero-failure upper bound

\[
1-\alpha^{1/n_{traj}}
\]

作为保守 upper floor；该 bound 约束 trajectory-failure，因 `P_cycle <= P_trajectory-fail` 也保守覆盖 cycle rate。结果同时保存 method、bootstrap SE 与 zero-event bound，不只输出 point estimate。

## 5. Production run

配置：seed `2026071413`，10,000 trajectories × 100 rounds，base `sigma_q/sigma_p=0.12/0.14 lambda`、`rho=0.25`、`loss_gamma=0.01`、outlier probability `5e-4`、scale 4；额外 rare `burst_and_leakage` 的 true trajectory probability `1e-4`，simulation allocation 20%。

- weighted `P_L = 0.04702085`；
- 95% trajectory-cluster bootstrap CI `[0.04643899, 0.04756083]`；
- q/p rates `0.01857991 / 0.02946997`；
- normal conditional rate `0.04701875`（8,000 trajectories）；
- rare conditional rate `0.06802000`（2,000 trajectories）；
- rare stratum simulated burst/leakage cycles `7856/8005`；
- 1e5 minimum 与 1e6 target gates 均 PASS。

这些数值证明实现的计算/统计 contract，不是实验 LER 或装置 rare-event rate reproduction。

## 6. 反 demo 测试

`tests/test_fast_monte_carlo.py` 的 19 项测试覆盖：

1. workload、概率、持续时间、seed、source length/type 与 hazard failure；
2. 1e5 cycles output contract 和真实 1e6 cycles execution；
3. fixed/changed seed、cluster bootstrap replay 与 CI containment；
4. 200k single-round analytic Gaussian calibration；
5. zero-event nonzero conservative upper bound；
6. burst/leakage deterministic mixture unbiasedness 和 allocation invariance；
7. stratum weight/count/cycle budget；
8. background leakage logical faults；
9. multi-round recovery causal ablation；
10. q/p/any event count invariants；
11. machine JSON round-trip、scope flags 和 public exports。

剩余边界：rare mixture parameters 未设备校准；vectorized core 尚未与 Fock simulator 逐点交叉验证；host throughput 不是 target-board timing；importance/stratified variance reduction 的最优 allocation 尚未自动求解。以上进入 R-N033，并由 T2.2/T2.3/T2.4/T5/T6 正常任务承接。
