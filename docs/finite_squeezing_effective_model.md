# T2.2.1 有限 squeezing 分解式 effective noise model

**实现：** `physics/finite_squeezing_noise.py`  
**测试：** `tests/test_finite_squeezing_noise.py`  
**机器结果：** `docs/t2_2_1_finite_squeezing_validation.json`  
**证据层：** syndrome-level effective model；不是 Fock/master-equation 或装置标定

## 1. 为什么不能只写一个 `sigma_eff`

一次两正交分量观测被拆成：

\[
\boldsymbol x_{\rm phys}
=\boldsymbol x_{\rm ch}
+\boldsymbol n_{\rm data}
+\boldsymbol b_{\rm env},
\]

\[
\boldsymbol y_{\rm obs}
=\boldsymbol x_{\rm phys}
+\boldsymbol n_{\rm anc}
+\boldsymbol n_{\rm meas}.
\]

- `channel`：允许完整 2×2 covariance 和 q-p correlation；
- `data_gkp`：data GKP 隔离 peak 的 finite-squeezing variance；
- `ancilla_gkp`：辅助 GKP peak 的观测注入，不能写进 physical data truth；
- `measurement`：classical analog/readout covariance；
- `finite_energy_envelope`：由 comb 格点分布与 peak-center contraction 得到的非高斯 shift。

因此 analytic budget 分开保存
`channel/data_gkp/ancilla_gkp/measurement/finite_energy_envelope`，并给出：

\[
\Sigma_{\rm phys}
=\Sigma_{\rm ch}+\Sigma_{\rm data}+\Sigma_{\rm env},
\]

\[
\Sigma_{\rm obs}
=\Sigma_{\rm phys}+\Sigma_{\rm anc}+\Sigma_{\rm meas}.
\]

代码同时输出 physical truth、observed analog、centered wrapped syndrome、实际 correction
和 corrected residual，避免把观测噪声错误写成 data state。

## 2. Peak variance 与已有 finite-energy 态族的连接

T1.2.1 的 damped-projector family 使用

\[
N_\Delta=\exp(-\Delta^2 a^\dagger a).
\]

Mehler kernel 给出 amplitude variance `tanh(Delta^2)`；隔离 peak 的 probability
variance 因而是

\[
\sigma_{\rm peak}^2(\Delta)
=\frac{\tanh(\Delta^2)}{2}.
\]

data 和 ancilla 都从该公式取 variance，但分别进入 physical 与 observation lane。direct
tests 又逐点构造 `damped_projector_state`，确认实现值与 state object 的
`amplitude_variance/2` 在浮点精度内一致；没有另造一套 Delta 约定。

这里的 Delta 仍是仓库 operational convention。它没有被静默换算成某篇论文的 dB、
vacuum variance 或 photon number；真实 normalization mapping 仍受 `PC-N01` gate 约束。

## 3. Envelope 不是第二个手写 Gaussian

对 damped-projector isolated-peak approximation，理想格点 `m lambda` 的 amplitude
coefficient 与收缩中心分别为

\[
c_m\propto
\exp\left[-\frac{\tanh(\Delta^2)(m\lambda)^2}{2}\right],
\qquad
q_m=\operatorname{sech}(\Delta^2)m\lambda.
\]

effective lattice-index probability 使用 `|c_m|^2`：

\[
p_m\propto\exp[-\tanh(\Delta^2)(m\lambda)^2],
\]

envelope deformation shift 为

\[
b_m=[\operatorname{sech}(\Delta^2)-1]m\lambda.
\]

代码保存完整离散 `p_m,b_m`，而不是把它高斯化后只留 variance。默认 `all` 表示
logical-state-averaged effective comb；`even/odd` 可做 parity sensitivity。截断使用离散
Gaussian 尾部的保守积分上界，并记录 captured-weight lower bound。

该项忽略相邻 peaks 的 coherent overlap，所以明确标为
`isolated_peak_incoherent_envelope_effective_model`。coherent/Fock 结论必须等 T2.3，不能
由本模型升级。

## 4. Correction 与 logical outcome

模拟器执行：

\[
\boldsymbol s
=\boldsymbol y_{\rm obs}
-\lambda\left\lfloor
\frac{\boldsymbol y_{\rm obs}}{\lambda}+\frac12
\right\rfloor,
\qquad
\boldsymbol c=\boldsymbol s,
\]

\[
\boldsymbol r
=\boldsymbol x_{\rm phys}-\boldsymbol c.
\]

logical parity 由 corrected physical residual 的 cell index 判定。测试独立用 `floor`
重算 syndrome、correction、residual 与 parity，逐样本完全一致。ideal diagonal-Gaussian
case 另与 T1.1.1 periodic analytic logical probability 比较，50 万样本结果在 5 SE 内，
证明不是只对 covariance 表做自洽检查。

## 5. 25 万样本×6 点 high-squeezing sweep

production config 固定：

- channel q/p sigma：`0.14 lambda / 0.10 lambda`，correlation `rho=0.2`；
- data Delta：`(0.50,0.42)`；ancilla Delta：`(0.36,0.32)`；
- measurement q/p sigma：`0.03 lambda / 0.025 lambda`；
- scales：`1,0.75,0.5,0.25,0.1,0`；
- 每点 250,000 samples；seed `2026071421`；
- 各点使用相同 component SeedSequence 分流，减少 endpoint 差分噪声。

| scale | finite-squeezing excess trace | observed trace | empirical/analytic covariance relative error | `P_L(any)` | 95% Wilson CI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 0.327704 | 0.523268 | 0.004473 | 0.031260 | [0.030585, 0.031949] |
| 0.75 | 0.184922 | 0.380486 | 0.004554 | 0.010484 | [0.010092, 0.010891] |
| 0.50 | 0.082278 | 0.277842 | 0.004256 | 0.002716 | [0.002519, 0.002928] |
| 0.25 | 0.020575 | 0.216139 | 0.003450 | 0.000792 | [0.000689, 0.000910] |
| 0.10 | 0.003292 | 0.198856 | 0.002808 | 0.000440 | [0.000365, 0.000530] |
| 0.00 | 0 | 0.195564 | 0.002456 | 0.000424 | [0.000351, 0.000513] |

结果满足：

1. analytic finite-squeezing excess 随 scale 严格下降；
2. `scale=0` 时 data/ancilla/envelope covariance 逐元素严格为零；
3. ideal endpoint 的 `physical_total=channel`、`observed_total=channel+measurement` 是精确
   等式，不是“差不多接近”；
4. broad finite-squeezing 与 ideal 的 logical-error Wilson intervals 明确分离；
5. 六点 observed covariance 的最大经验/解析相对误差为 `0.004554`。

## 6. 反简化检查

| 风险 | 实际检查 |
| --- | --- |
| 四类噪声仍被内部合并 | 每类 component array/covariance 独立保存，physical/observed 只在最后组合 |
| Delta 与旧态族脱节 | 逐点对 `FiniteEnergyGKPState.amplitude_variance/2` |
| envelope 只是另一个 Gaussian | 保存离散 lattice-index weights 和 non-Gaussian shifts；even/odd sensitivity 有测试 |
| covariance 公式自证 | 26 万样本逐 component 经验 covariance；ideal case 再对独立 periodic analytic LER |
| high-squeezing 只测一个小 Delta | 六点 sweep，末点使用显式 `Delta=0` ideal contract |
| ablation 改变随机数导致不可归因 | SeedSequence 分流；关闭 envelope 后 channel/data/ancilla/measurement arrays 逐样本不变 |
| 把 measurement 写进 physical state | `physical` 与 `observed` 分层，重组恒等式逐样本测试 |
| 冒充实验/Fock 模型 | JSON 和 API scope 均禁止 device-calibrated/Fock/lifetime claim |

## 7. 当前边界与后续接口

- data/ancilla peak 采用独立轴 Gaussian isolated-peak law；更一般的 coherent、rotated 或
  protocol-round-dependent backaction 尚未实现；
- envelope 使用 logical-state-averaged incoherent lattice mixture；不等同 density matrix；
- 当前是单轮 standard wrapped correction，不替代 T2.1 多轮 recovery/leakage dynamics；
- readout misclassification、ancilla bit/phase flip、reset failure 和 protocol-specific
  information flow 属 T2.2.2；
- DAC/AWG、virtual rotation、latency 和 displacement error 属 T2.2.3；
- Fock/coherent 与跨保真度校准属 T2.3.1--T2.3.3。

允许论文写“the effective simulator separately accounts for channel, data-GKP peak,
ancilla/measurement and finite-energy-envelope contributions, and recovers the ideal endpoint
as squeezing increases”。禁止写真实 squeezing dB 已校准、完整 approximate-GKP recovery、
实验 logical lifetime 或 Fock fidelity 已验证。
