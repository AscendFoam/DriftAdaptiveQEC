# T2.3.4 短时域可微 sBs trajectory simulator

**日期：** 2026-07-14  
**状态：** Implemented and validated  
**实现：** `physics/differentiable_sbs_trajectory.py`  
**机器结果：** `docs/t2_3_4_differentiable_trajectory_validation.json`、`docs/t2_3_4_differentiable_trajectory_validation_cuda.json`

## 1. 任务边界

本实现服务于 T2.3.5--T2.3.7 的 Feedback-GRAPE 可行性门。它显式演化
finite-cutoff cavity 与二能级 ancilla 的联合密度矩阵，支持随机或强制的 `g/e`
trajectory、可微门参数、历史条件 policy、trajectory probability、final-state reward
与资源画像。

它不是：

- pulse/ECD Hamiltonian 或真实 microwave waveform simulator；
- 多能级 transmon、`f/higher` leakage、SPAM 或装置标定模型；
- T2.3.5 的完整两项梯度 finite-difference 验证；
- T2.3.6 的 cutoff/batch/horizon 可行性 envelope；
- T2.3.7 的 standard/MF/NMF 排名；
- FPGA、实板或目标装置时序证据。

## 2. 一手来源合同

本地一手全文的四个直接锚是：

| 合同 | 本地锚 | 实现含义 |
| --- | --- | --- |
| physical gates | Puviani et al. 补充材料 411--434 行 | 使用 `R_q(phi,theta)`、`ECD_qc(beta)`、`D_c(alpha)`、`VR_c(theta_VR)` 的显式矩阵 |
| fixed timing | 451--459 行 Table S1 | `0.1+0.5+0.7+0.3+0.1+2.3+1.0=5 us`/half-cycle，`10 us`/full cycle |
| Feedback-GRAPE | 467--495 行 | 返回 `R(m)`、`log P_theta(m)`，为两项梯度保留计算图 |
| 15 controls | 511--527 行 Table S4 | 每 half-cycle 8 个 qubit-rotation、6 个 ECD 实/虚部和 1 个 VR；固定 layer-4 `alpha` |

这里的 `5 us` 是该数值论文的 Table S1 训练 profile。它与 Sivak Table S3 的
`4.924 us` constituent/`9.848 us` X+Z reference timeline 是两条不同 provenance，
也都不是目标低价 FPGA 的实测时序。

## 3. 数学与执行顺序

联合维数为 `2N`，其中 `N` 是 cavity Fock cutoff。每个 half-cycle 执行：

1. entering-cycle idle；
2. 三层 `R_q(phi_l,theta_l)` 后接 `ECD_qc(beta_l)`；
3. 第四层 `R_q(phi_4,theta_4)` 后接固定 `D_c(alpha)`；
4. measurement/reset 时间窗的 CPTP idle；
5. ancilla `g/e` 投影、按条件概率采样或 replay，并数值 reset 到 `|g>`；
6. `VR_c(theta_VR)` 与最后 idle。

门定义为

\[
D(\beta)=\exp(\beta a^\dagger-\beta^*a),
\]

\[
ECD(\beta)=D(\beta/2)\otimes |g\rangle\langle e|
+D(-\beta/2)\otimes |e\rangle\langle g|,
\]

\[
R_q(\phi,\theta)=\exp\{-i\theta(\sigma_x\cos\phi+\sigma_y\sin\phi)/2\},
\qquad
VR(\vartheta)=\exp(i\vartheta a^\dagger a).
\]

每个文献 timing interval 后的 noise 使用 exact finite-cutoff pure-loss Kraus、ancilla
amplitude damping 与 pure dephasing Kraus。默认 high-noise 训练参数为
`Ts/Tcycle=24.5`、`T1/Tcycle=5`、`T2/Tcycle=6`，即
`Ts=245 us`、`T1=50 us`、`T2=60 us`；这些仍是论文 scenario，不是项目装置参数。

第 `j` 个 outcome 的条件概率为

\[
p_j(m_j)=\operatorname{Tr}[P_{m_j}\rho_jP_{m_j}],\qquad
\log P_\theta(\mathbf m)=\sum_j\log p_j(m_j).
\]

对纯初态 target，reward 为

\[
\mathcal R(\mathbf m)=\langle\psi_0|\rho_c(T|\mathbf m)|\psi_0\rangle.
\]

采样 decision 对概率做 `detach`，但所选 conditional branch、`log P` 与 reward 不
detach；因此固定 trajectory 上的 reward path 与 score path 均保留 autograd graph。

## 4. 控制与 history policy

输入可以是预给定的 `[batch, half_cycles, 15]` raw correction，也可以是逐 half-cycle
调用的

```text
control_policy(previous_g_e_history, half_cycle_index) -> [batch, 15]
```

后一接口保证第 `j` 次控制只能看到前 `j` 次测量结果，不能偷看未来 outcome。raw
correction 经 `tanh` 映射到论文范围：前 14 项为 `[-2,2]`，VR 为 `[-1,1]`，再加到
Table S4 nominal control 上。layer-4 `alpha=sqrt(pi/2)` 不进入训练参数。

## 5. 输出与资源画像

每次运行返回：

- final joint/cavity density；
- `g/e` outcome sequence 与每一步 conditional probability；
- `log_probability` 和 `trajectory_probability`；
- final pure-target fidelity reward；
- 实际使用的 15 参数序列；
- trace/Hermiticity/PSD diagnostics；
- cutoff、joint dimension、batch/horizon、matrix exponential、unitary/CPTP 调用数、
  state bytes、autograd lower bound、host wall time 与可选 CUDA peak allocation。

`wall_time`/CUDA allocation 只描述当前 host 的一次短跑，不能替代 T2.3.6 的 warm-up、
repeat、cutoff/batch/horizon scan，更不能外推硬件 latency。

## 6. 验证结果

cutoff 8、batch 4、1 full cycle 的 17 个 production gates 在 CPU 与 CUDA 均通过：

| 指标 | CPU | CUDA |
| --- | ---: | ---: |
| 四条 `gg/ge/eg/ee` trajectory probability 和 | `1.0` | `1.0` |
| open-loop gradient norm | `0.9365002774` | `0.9365002774` |
| history-policy gradient norm | `1.042401137` | `1.042401137` |
| history-conditioned control separation | `1.066235432` | `1.066235432` |
| 最大 gate unitarity residual | `8.88e-15` | `7.62e-15` |
| 最大 idle completeness residual | `8.88e-16` | `8.88e-16` |
| 最大 trace error | `2.22e-16` | `0` |
| 最小 final density eigenvalue | `-1.07e-16` | `0` |
| 单次 wall time（非 benchmark） | `0.0503 s` | `0.528 s` |
| CUDA peak allocation | — | `11,536,384 B` |

CPU/CUDA 的 reward、trajectory probability 与两类 gradient norm 在 float64 rounding
内一致。小矩阵 CUDA 短跑包含 cold-start/kernel overhead，不能据此判断 GPU 较慢。

专门测试 `37 passed`，覆盖参数/时序负路径、SciPy 独立 matrix exponential、gate
unitarity、21 个 CPTP completeness gate、四分支归一化、3000-shot sampling
calibration、seed/replay、两条 gradient graph、history causality、两周期 trace/PSD、
资源计数和 CUDA smoke。minimal recovery Python 无 PyTorch 时该文件 module-skip，既有
NumPy 路线不被伪装成 torch fallback。

## 7. 运行方式

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m physics.differentiable_sbs_trajectory --device cpu --cutoff 8
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m pytest tests\test_differentiable_sbs_trajectory.py -q
```

第二条环境变量只绕开本机 DLEnv 中残缺的第三方 Hydra/OmegaConf pytest plugin，
不改变仿真代码或依赖。
