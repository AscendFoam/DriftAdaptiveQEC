# T2.1.1 连续漂移与离散 regime 的 syndrome stream generator

**日期：** 2026-07-14  
**实现：** `physics/syndrome_stream.py`  
**协议：** `PROTO-SBS-MAIN`  
**证据范围：** protocol-aligned mixed-state syndrome-level effective model；非 cavity--transmon/Fock/device-calibrated simulator

## 1. 目标与因果顺序

生成器逐周期完整接收 `DriftState`，按固定因果顺序执行：

1. 读取 `mu_q/mu_p/sigma_q/sigma_p/rho/p_outlier/outlier_scale`，从相关二分量 Gaussian mixture 采样 channel displacement；
2. 读取 `loss_gamma`，以 `eta=exp(-loss_gamma)` 衰减上周期 physical residual，并加入方差 `(1-eta)V_env` 的显式 loss-environment noise；
3. 使用共享 `LATTICE_CONST=sqrt(2*pi)` 做 standard binning，分别得到 true folded syndrome、lattice index 和 q/p parity；
4. 加入独立 measurement noise，输出 analog syndrome，并再次 wrap 成半开区间 residual syndrome；
5. 根据 folded magnitude 生成 coarse recovery depth，根据 loss 与 `burst_active` 生成 leakage hazard；
6. 输出按执行顺序 `(X,Z)` 的 `g/e/leakage`，更新恢复深度、单象限 residual 与 run length；
7. 累积 q→logical-X、p→logical-Z parity，输出 `I/X/Z/Y` hidden truth。

`step/time/source/regime/seed/event_id` 不参与额外随机律，但原样保存在每步 `drift_state` truth provenance；因此没有把未用字段静默丢弃。`source/regime/event_id` 仍是 hidden metadata，不能进入 deployable feature。

## 2. 数学约定

第 t 周期在测量前的位移为

\[
r_t^- = \sqrt{\eta_t}\,r_{t-1}^+ + d_t + \xi_t,
\qquad
\xi_t\sim\mathcal N(0,(1-\eta_t)V_{env}I),
\]

其中 `d_t` 直接由 `DriftState` 的 core/outlier mixture 采样。这里选择“先 loss、后本周期 channel displacement”的 effective ordering；它不是把 loss 偷换成旧 `gamma/2` scalar-sigma proxy。

对每个 quadrature，shared standard-binning 计算

\[
n=\left\lfloor r^-/\lambda+1/2\right\rfloor,
\qquad
s=r^- - n\lambda\in[-\lambda/2,\lambda/2),
\]

并用 `n mod 2` 更新 logical Pauli truth。measurement noise 只改变 observed analog/residual，不回写 true logical label。

恢复深度是 syndrome-level coarse state，不是 T2.0.2 的装置标定 transition probability。成功动作只缩小被选 constituent 的 residual：X 动作只改 q，Z 动作只改 p；未受影响轴必须保持。泄漏会增加 depth，并在首次出现时建立 pending X/Z quadrature，保证泄漏结束后不会卡在“depth>0 但无恢复动作”的伪状态。

## 3. observed/truth schema 隔离

`ObservedSyndromeStep.as_deployable_dict()` 只包含：

- cycle/drift step、time；
- analog q/p、wrapped residual q/p；
- X/Z `g/e/leakage`；
- `(0,pi/2)` quadrature phases；
- observed X-e、Z-e 与 leakage run length；
- validity 与 observation scope。

它不含 `DriftState`、hidden regime、outlier component、leakage kind/hazard、recovery depth、logical truth 或 simulator residual。`SyndromeTruthStep` 与 read-only `truth_records()` 单独保存这些字段。T2.1.2 才会建立正式多轮 controller memory；本任务的 run length 只是 stream observation，不冒充完整控制器。

## 4. 参数来源与禁止外推

默认 `F_g=0.9997`、`F_e=0.9914` 只复用本地一手文献中可安全锚定的对角 readout 值。默认 depth law、recovery probability/gain、loss/burst leakage hazard、higher-state duration与 perfect leakage classification 都是项目 assumptions；没有从两个对角 fidelity 猜完整 4×3 confusion matrix。

因此当前结果只允许用于快速 syndrome-level synthetic benchmark、schema/causality 测试和后续 controller/Monte Carlo 接口。禁止把它写成：

- cavity--transmon Hamiltonian 或 Fock-space 演化；
- 装置校准的 sBs transition/readout/reset/leakage 模型；
- ADC/IQ classifier、真实 microwave waveform 或目标 FPGA timing；
- 实验 logical lifetime、occupancy 或 leakage rate 的定量复现。

T2.2/T2.3 负责 finite-energy、ancilla、Fock 与跨保真度校准；T2.4/T6 负责 timing/hardware evidence。

## 5. 反 demo 验证

`tests/test_syndrome_stream.py` 的 21 项测试覆盖：

1. 参数概率、几何持续时间、seed、输入类型与 step contract 的 fail-closed 边界；
2. 空 stream、公开惰性导出和固定 protocol/model scope；
3. required observed/truth fields 与 deployable schema hidden-key prohibition；
4. analog→residual 半开区间恒等式；
5. q→X、p→Z 及 `X*Z*X*Z=I` logical composition；
6. 20,000 周期相关 Gaussian mixture 的 mean/covariance/outlier-rate 统计复核；
7. loss 对 carried residual 的 `sqrt(eta)` 衰减；
8. 8,000 周期 loss-environment variance 与 leakage hazard 复核；
9. hazard 超界拒绝、burst-only leakage、f 与 higher persistence；
10. 泄漏后 pending quadrature 恢复，防止 recovery-depth deadlock；
11. X/Z constituent-specific residual correction 负控；
12. 5,000 周期 configured g/e confusion 频率复核；
13. recovery depth、e/leakage runs 的因果更新；
14. same-seed exact replay、长短 run prefix stability 与 changed-seed divergence。

相邻 `DriftState + T2.0.1--T2.0.6 + T2.1.1` 回归在 protocol contract 同步后为 `161 passed`。完整 `tests/` 的旧 R-N012 失败需与本任务结果分开报告。
