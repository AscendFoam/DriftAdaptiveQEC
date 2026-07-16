# T2.2.3 控制链与 active-correction imperfection

**日期：** 2026-07-14  
**状态：** Implemented effective layer  
**实现：** `physics/control_imperfections.py`  
**机器证据：** `docs/t2_2_3_control_imperfection_validation.json`

## 1. 实现结论

本任务把 controller request、数字硬件 command 和 hidden physical realization 分成
三层，没有用单个 additive Gaussian 冒充完整控制链。默认因果顺序为：

1. requested Cartesian correction 转成 AWG amplitude/phase；
2. AWG amplitude/phase 分别量化，转回 I/Q；
3. I/Q 经过 signed DAC quantization 和 saturation；
4. affine pulse gain/crosstalk/bias 形成 mean physical displacement；
5. latency drift/diffusion 在动作到达前累积；
6. active displacement 加入 command-dependent multiplicative gain error 和 full
   covariance additive error；
7. virtual-rotation command 单独量化，再加入 gain/bias/noise；
8. 按显式 `action_order` 计算非交换的 displacement/rotation 后物理 residual。

该实现是 sBs 主数字孪生的 control-imperfection effective layer。所有 bit width、
full scale、matrix、covariance、latency law 和 calibration error 必须显式传入并带
provenance；没有目标 DAC/AWG 或微波链的 device-calibrated 默认值。

## 2. 现有项目接口复用

- 继续使用 T2.1.2/现有 fast-loop 的符号约定：
  `post_action_residual = pre_action_residual - correction`；
- direct test 把现有 `LinearRuntime(Q4.20)` 的 `correction_applied` 接入本层
  `ControlActionRequest`，因此本层是 Q4.20 decoder **之后** 的物理输出链，不重复
  把 decoder arithmetic 当 DAC；
- T2.0.4 的 virtual-rotation stage 与 T2.2.2 的 misclassification-triggered
  rotation 保持独立来源：本任务处理 command quantization/calibration/noise；
- `ControlActionRecord` 保存 request、AWG/DAC codes、dequantized command、
  saturation、latency 和 bank version；`ControlActionTruth` 才保存 physical
  displacement、diffusion、gain error、rotation error 和最终 residual。

因此 T2.1.2 中的“实际执行 correction”在此层应解释为 controller/DAC 已发出的
dequantized command，而不是控制器可以观测的真实 cavity displacement。

## 3. 量化约定

### 3.1 AWG

- amplitude 使用 unsigned `0 ... 2^B-1` codes，覆盖
  `[0, amplitude_full_scale]`；
- phase 使用 `2^B` 个周期 codes，按 `2pi/2^B` 量化；
- Cartesian request 先转 polar，再由量化后的 polar command 转回 I/Q。

### 3.2 DAC

I/Q 各自使用 two's-complement-style signed codes：

`-2^(B-1) ... 2^(B-1)-1`，

step 为 `full_scale/2^(B-1)`。超量程先记录 saturation，再输出 clipped code。
rounding 使用 NumPy round-to-nearest-even；这只是软件 reference convention，不能
在没有 RTL/板卡证据时声称目标 DAC 也采用相同规则。

### 3.3 Virtual rotation

virtual rotation 有独立 phase code。其 physical angle 为

`theta_actual = gain * theta_code + bias + epsilon_theta`，

而物理 frame error 为 `theta_actual - theta_requested`。因此 quantization、
systematic calibration 和 stochastic jitter 可分别审计。

## 4. Pulse 与 active displacement

dequantized I/Q command `u` 先经过

`u_mean = G u + b`，

其中 `G` 允许 amplitude gain、axis imbalance 和 q/p crosstalk，`b` 是
systematic offset。随后

`u_actual = (1 + epsilon_g) u_mean + epsilon_a`，

`epsilon_g` 是 scalar multiplicative error，`epsilon_a` 允许 full 2×2
covariance。于是 active-displacement covariance 含

`sigma_g^2 u_mean u_mean^T + Sigma_active`，

不会错误地把 command-dependent gain noise 固定成与 action 无关的 isotropic noise。

## 5. Latency-induced noise

动作到达前：

`r_action = r_before + v_latency * t + epsilon_latency`，

`epsilon_latency ~ N(0, t * Sigma_latency_per_us)`。

`latency_us` 是 request/record 的可见调度量；随机 diffusion realization 只属于
truth。超出 `max_latency_us` 会 fail closed。本层表征 latency 导致的物理漂移/
扩散，不声称 T4.2/T4.3 的 deadline suppression、fallback、CRC、stale 或 atomic
commit 已实现。

## 6. 非交换动作顺序

实现同时支持：

- `displacement_then_virtual_rotation`：
  `r_out = R(delta_theta) (r_action - u_actual)`；
- `virtual_rotation_then_displacement`：
  `r_out = R(delta_theta) r_action - u_actual`。

两者在一般情况下不相等。direct test 的 `pi/2` fault case 得到
`(0,0.8)` 与 `(-0.2,1.0)`，防止代码把顺序差异静默消掉。

## 7. 解析矩与 vectorized sampler

对固定 request，本模块给出 exact first/second moments。对 Gaussian angle error，
`E[R(theta)] = exp(-sigma_theta^2/2) R(mu_theta)`，二阶矩用
`E[cos(2theta)]` 和 `E[sin(2theta)]` 闭式计算。两种 action order 都有独立
公式。

`sample_fixed_request` 使用四个独立 SeedSequence 子流分别生成 latency、
multiplicative gain、active additive 和 rotation noise。改变 batch size 时前缀逐
component 完全相同，适合 ablation，不会因开关某一项而重排所有随机数。

## 8. Production 验证

100,000-sample combined scenario：

| 指标 | 结果 |
| --- | ---: |
| 最大 empirical/analytic mean z-score | 1.5358 |
| covariance relative Frobenius error | 0.001697 |
| pulse systematic displacement error norm | 0.006944 |
| virtual-rotation systematic error | 0.002798 rad |
| exact ideal endpoint max residual | 0 |

bit-resolution sweep 使用 12 radii × 73 phases：

| shared bits | displacement RMS quantization error |
| ---: | ---: |
| 6 | 0.041179 |
| 8 | 0.010172 |
| 10 | 0.002460 |
| 12 | 0.000627 |

latency-only analytic covariance trace：

| latency (us) | covariance trace |
| ---: | ---: |
| 0 | numerical zero |
| 2 | 0.000360 |
| 5 | 0.000900 |
| 10 | 0.001800 |

7 项 production gates 全 PASS。

## 9. 反简化测试

`tests/test_control_imperfections.py` 共 33 项：

- AWG unsigned、periodic phase 与 signed DAC code-level exact cases；
- saturation、zero command、request/command preservation；
- affine pulse/crosstalk、multiplicative rank-one covariance、full additive covariance；
- latency drift order 与 50,000-sample covariance scaling；
- virtual rotation 与两种 action-order negative control；
- 两种 order 各 80,000 samples 的 exact analytic moment calibration；
- ideal endpoint、seed replay、batch prefix/read-only；
- sequential physical residual carry；
- deployable/truth schema isolation；
- Q4.20 fast-loop integration；
- 100,000-sample production validation、JSON、非法 config/request/latency。

## 10. Claim boundary

允许声称 causal command encoding、pulse/latency/active-action sensitivity 和解析矩已
实现。禁止声称：

- 目标 DAC/AWG bit depth、full scale、rounding 或 microwave fidelity 已实测；
- 文献 `1e-3` DAC sensitivity 已迁移为本项目 random-walk rate；
- host latency law 等同于 hard-real-time board timing；
- T4.2/T4.3 fallback/deadline/atomic update 已闭环；
- 该 effective residual 等同于 Fock/master-equation logical channel。
