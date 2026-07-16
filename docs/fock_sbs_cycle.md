# T2.3.2 Fock-space 协议对齐 SBS 一轮

**实现：** `physics/fock_sbs_cycle.py`  
**机器验证：** `docs/t2_3_2_fock_sbs_cycle_validation.json`  
**一手公式：** Sivak *et al.*, *Real-time quantum error correction beyond break-even*, Supplementary Information “Small-Big-Small (SBS) protocol”，本地 `relative_papers/Real-time_quantum_error_correction_beyond_break-even/` 与 [arXiv:2211.09116](https://arxiv.org/abs/2211.09116)  
**边界：** finite-cutoff、single-oscillator、analytic-Kraus reference；不是 ECD pulse Hamiltonian、显式 transmon、多能级 reset、装置校准或硬件结果。

## 1. 本任务真正实现的对象

T2.3.2 不再把“一轮纠错”近似成 modular POVM、人工 half-cell feedback 和额外理想逻辑门的拼装。主路径直接实现论文给出的 constituent Kraus operators：

```text
K_g^X = cos(sqrt(pi) p) cos(sqrt(pi) Delta^2 x)
        + sin(pi Delta^2/2) cos(sqrt(pi) p)

K_e^X = -cos(pi Delta^2/2) sin(sqrt(pi) p)
        + i cos(sqrt(pi) p) sin(sqrt(pi) Delta^2 x)
```

其中 `x=(a+a^dagger)/sqrt(2)`、`p=i(a^dagger-a)/sqrt(2)`；Z constituent 严格按
`(x,p)->(-p,x)` 替换。完整一轮按时间执行 X 再 Z，矩阵标签保持论文的
`K_z^Z K_x^X` 顺序，因此 chronological `(X,Z)=(e,g)` 对应 `K_ge`。

这些 Kraus 已包含 SBS 的自主 backaction 与确定性逻辑翻转。实现不会再叠加一个
“理想逻辑 X/Z lift”，只在 classical Pauli frame 中分别记录 X constituent 和 Z
constituent 的 deterministic frame update。

## 2. 坐标归一化桥

仓库 `finite_energy_gkp.py` 的 operational q 坐标以 `sqrt(2*pi)` 为 logical-cell spacing；
论文 canonical oscillator 坐标 `x` 的 logical spacing 为 `sqrt(pi)`。因此本任务显式使用

```text
q_repo = sqrt(2) x_canonical
psi_canonical(x) = 2^(1/4) psi_repo(sqrt(2) x).
```

`fock_density_model.project_finite_energy_gkp` 新增
`source_coordinate_scale`，按 `sqrt(scale) psi_source(scale*q)` 做保持归一化的 dilation。
T-RISK-20260714-01 后，标准 damped-projector Fock preparation 固定
`scale=sqrt(2)`，并同时缩放 peaks、width、envelope 与 Jacobian；`scale=1` 只能通过低层
generic projection 明确调用。直接测试证明 decoder wavefunction 与 canonical source 的
Fock coefficients 等价，防止“只移动中心”的伪 bridge。

## 3. 有限截断的 CPTP completion

论文公式在无限维 Weyl algebra 中定义。有限 Fock 截断满足
`[a,a^dagger] != I` 的 top-boundary correction；直接对截断 `x,p` 做 matrix function 后，
原始 pair 一般不再满足 `sum_b K_b^dagger K_b=I`。这不是 OCR 缺项：本任务从官方
arXiv TeX 源重新核对了公式。

为得到可执行的有限维 quantum instrument，对每个 X/Z pair 单独执行共享右侧 completion：

```text
G = sum_b K_b^dagger K_b
K_b_completed = K_b G^(-1/2).
```

这等价于对整个 Kraus pair 做最小共享 right tightening，不逐 outcome 偷做独立归一化。
实现同时保留并输出：

- raw full-space Frobenius/operator completeness error；
- raw code-subspace completeness error；
- `G` 的最小/最大特征值和条件数；
- completion 前后 pair 的 Frobenius change；
- completed completeness error。

production `N=24, Delta=0.34` 中，X/Z raw Frobenius error 均为 `1.091934`，raw
code-subspace error 均为 `0.016552`，`G` 谱为
`[0.887803,1.433384]`、条件数 `1.61453`；completion pair change 为 `0.507369`，
completed error 降至 `1.11e-14/1.36e-14`。因此下游结果必须写成
“finite-cutoff completed analytic SBS map”，不得假装 raw 截断天然 CPTP。

## 4. 一轮因果顺序与 schema

每轮执行：

```text
logical initialization
-> idle channels
-> hidden X Kraus g/e
-> noisy observed X g/e
-> observed-routed residual virtual phase action
-> X frame update
-> hidden Z Kraus g/e
-> noisy observed Z g/e
-> observed-routed residual virtual phase action
-> Z frame update
-> code-space logical projection and frame correction.
```

ideal SBS/象限切换的 classical virtual action 已包含在解析 constituent map 和
`(x,p)->(-p,x)` 定义中。显式 `controller_residual_phase_by_observed` 只表示额外的
观测分支残余相位 scenario，用于模拟 readout misclassification 后的 controller action
fault；默认精确为零，不冒充论文装置校准角。

hidden quantum outcome、hidden probability、Kraus 后 code support 只在 truth record；
controller-visible record 仅含 observed outcome、顺序、执行的残余相位与 frame。
readout confusion 四个元素均非零时，完整 hidden `(g/e)^2` × observed `(g/e)^2`
instrument 有 16 条 branch；perfect readout 时退化为四条物理 `K_gg/K_ge/K_eg/K_ee`
路径。某些条件 branch 对 code space 的支撑数值为零，此时 branch projection 明确为
`None`，不会伪造一个归一化 logical state；unconditional projection 仍有定义。

## 5. 生产验证

`docs/t2_3_2_fock_sbs_cycle_validation.json` 的 16 个 gate 全部通过：

- direct T2.3.1+T2.3.2：`99 passed`；
- clean 六个 Pauli eigenstates：条件 logical fidelity `0.999953`、code survival
  `0.969508`、code-weighted fidelity `0.969463`；
- 加入 displacement/loss/thermal/phase/Kerr/high-Fock stress、readout confusion 和
  observed-routed residual phase 后：条件 fidelity `0.999855`、survival `0.899286`、
  code-weighted fidelity `0.899156`，没有出现“注册噪声反而改善”伪结果；
- `a` 与 `a^dagger` 首层 photon-error state 的单轮平均 code-survival 增益
  `0.518107`，直接验证 trickle-down 的纠错方向；
- 100,000 次 branch sampling 相对 exact probabilities 的最大 z-score `1.8907`；
- 16 条受测 branch 的最小 eigenvalue `1.61e-9`，总概率误差 `2.22e-15`；
- `N=18/24/30/36/42` cutoff sweep 的条件 fidelity 为
  `0.999066/0.999953/0.999990/0.99999938/0.99999918`；最后两点差约 `2.00e-7`；
- survival 随该 sweep 上升，最后两点为 `0.978439/0.979209`、差 `0.000770`；
  这仍只是有限五点数值稳定性，不写成无限维收敛定理。

独立单测逐矩阵重构 X 公式和 Z substitution，并检查 raw defect、共享 completion、
CPTP、idle 次序、hidden/observed schema、读出完全翻转、观测分支 action、四/十六分支、
概率 mixture、Pauli frame、photon loss/gain、seed replay、非法参数和 scope fail-close。

## 6. Claim 边界与后续

当前允许：finite-cutoff completed analytic SBS one-round reference、X/Z Kraus-order
alignment、hidden/observed branching、scenario classical residual action、Pauli-frame 和
code projection、cutoff/positivity/completeness diagnostics。

当前禁止：无限维 exact channel、pulse-level ECD、explicit ancilla/transmon Hilbert space、
真实 `g/e/f` reset dynamics、device-trained RL parameters、装置 calibrated logical lifetime、
实验 break-even 或 FPGA/hardware fidelity。T2.3.3 将把本 Fock reference 与 effective、
noise-transfer surrogate、syndrome model 做跨 fidelity 趋势与失效区归因；该工作已由
T2.3.3 完成，坐标修正由 T-RISK-20260714-01 回灌。
