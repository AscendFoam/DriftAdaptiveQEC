# T3.2.2 连续漂移 EWMA / Kalman adaptive MAP

## 1. 结论

T3.2.2 建立了三个 observed-only、one-window-delay 的传统自适应 periodic-MAP baseline：

- `latest_window_periodic_moment_map`：只用上一窗口的二维周期矩估计；
- `ewma_periodic_moment_map`：递推四个复特征的 EWMA；
- `kalman_constant_velocity_periodic_map`：在均值、完整二维协方差及其速度上做 constant-velocity Kalman 预测。

在 4 类连续 synthetic wrapped-Gaussian drift、8 个独立 evaluation seeds、1,572,864 个配对
evaluation samples 上，formal static MAP 减 EWMA 的 aggregate LER gain 为
`0.009900 [0.009622,0.010179]`，减 Kalman 的 gain 为
`0.009933 [0.009658,0.010208]`。区间是 evaluation-seed cluster 上的双侧 Student-t 95% CI。
四个场景的两个 adaptive gain CI 下界均为正，oracle 仍保持严格参考下界。

这些结果只证明：在已登记的连续二维 wrapped-Gaussian syndrome-level 场景中，EWMA/Kalman 是
必须保留的强传统 baseline。它们不证明 CNN 优势，也不覆盖 loss、outlier、leakage、有限能协议、
设备标定或 FPGA 综合/实测。

## 2. 为什么不能直接复用 T1.3.4

旧 `adaptive_drift_alignment.py` 只覆盖单一 step drift 和一个 evaluation seed，Window/EKF 又把
协方差压成固定轴比的 `(sigma, theta)` 子空间。T3.1.2 引入 training-average formal static 后，旧
EKF 已不能显著优于 static。因此本任务没有把旧 alignment 结果改名包装，而是重新冻结连续漂移、
完整协方差、训练/评估隔离、相同 observation/update budget 和多 seed 配对统计合同。

## 3. 周期矩与完整协方差

令 `L = sqrt(2*pi)`、`k = 2*pi/L`。每个窗口只消费折叠到 `[-L/2,L/2)` 的
`residual_q/residual_p`，计算四个圆特征：

```text
phi_q   = mean(exp(i*k*q))
phi_p   = mean(exp(i*k*p))
phi_q+p = mean(exp(i*k*(q+p)))
phi_q-p = mean(exp(i*k*(q-p)))
```

相位给出周期均值，模长通过 wrapped-Gaussian characteristic function 给出 `var(q)`、`var(p)`、
`var(q+p)` 和 `var(q-p)`；两个 joint variance 分别恢复正、负相关信息。最后做 shrinkage、相关系数
裁剪和 SPD 投影。这样避免用线性均值处理 half-cell 边界，也避免把 `q/p` 错当独立。

## 4. 两个递推 baseline

EWMA 在四个复特征上执行 `z_t = alpha*y_t + (1-alpha)*z_(t-1)`，再从平滑后的特征恢复完整
Gaussian 参数。Kalman 使用 10 维状态：

```text
(mu_q, mu_p, log(var_q), log(var_p), atanh(rho))
+ 对应五个 constant-velocity 分量
```

均值测量先围绕当前预测做 nearest periodic lift，协方差经 log/atanh 变换保证正定/有界；更新采用
Joseph covariance form。Kalman 输出中的 circular features 由预测 Gaussian 解析重建，不以 NaN
占位。两种方法都先用上一状态解码当前 evaluation buffer，再用当前独立 observation buffer 更新，
因此没有 same-window leakage。

## 5. 公平性与调参

- training seeds：`20260811--20260813`；evaluation seeds：`20260831--20260838`，严格不相交；
- 每场景 48 窗，前 4 窗 calibration；每窗 384 个 observed residuals、1024 个独立 paired
  evaluation samples、一次 update、一个窗口因果延迟；
- static 参数只从 frozen training states 拟合；
- EWMA/Kalman 超参数只在 12 条 materialized training traces 上按 observation-only、独立下一窗口
  periodic-moment forecast score 选择；
- 扩展后的完整网格选择 `alpha=0.85`、`process_scale=2.0`、`measurement_scale=0.75`，三者均不是
  候选边界；evaluation 结果不参与选择；
- standard、static、window、EWMA、Kalman 与 full-state oracle 读取相同 evaluation buffers，逐 seed
  trace SHA-256 唯一。

## 6. 生产结果

| 场景 | standard | static | latest window | EWMA | Kalman | oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear mean | 0.006170 | 0.007052 | 0.000944 | 0.000946 | 0.000944 | 0.000900 |
| variance/correlation ramp | 0.013603 | 0.011831 | 0.004125 | 0.004082 | 0.004066 | 0.003944 |
| sinusoidal joint | 0.023659 | 0.019475 | 0.004316 | 0.004397 | 0.004300 | 0.003698 |
| smooth mixed | 0.019117 | 0.017166 | 0.006543 | 0.006498 | 0.006482 | 0.006139 |

`latest window` 在部分场景与递推法统计上相当，因此没有伪造“递推法必须显著胜过 window”的门。
正式门要求：每个场景至少一个 recursive point 不差于 latest-window point，且 recursive family 同时
改善 mean 与 covariance tracking。四场景均通过；adaptive NLL/Brier 也均优于 static。

## 7. 反简化检查

1. 从 step/单 seed/fixed-axis-ratio 旧实现升级为 4 类 smooth drift、8 evaluation seeds、完整协方差。
2. 特征估计用 `q+p/q-p` 联合矩，并以正负相关 Monte Carlo 和 lattice-translation invariance 测试，
   不用 q/p factorized demo。
3. latest-window、EWMA、Kalman 三者全部保留；`alpha=1` 与 latest-window 等价有直接测试。
4. Kalman 检查边界 unwrap、SPD covariance、finite gain、线性均值 anticipation；非法 shape、NaN、
   越界和短窗口 fail closed。
5. 首轮调参最优落在网格边界后扩大候选网格并重跑，正式选择落在内部；不把 boundary optimum
   当成已充分搜索。
6. 置信区间以 evaluation seed 为 cluster，不把窗口/样本伪装成独立重复。
7. 所有受 comparison registry hash 影响的 T3.1.1--T3.2.1 正式 artifacts 均按最终注册表重生成。

## 8. 成本与证据边界

每窗口 384 个 observation、一次 update。周期矩每 observation 的确定性代理为 2 个 complex
exponentials 和 2 个 complex products；EWMA 保存 4 个 complex states；Kalman 保存 10 个 states、
100 个 covariance values，innovation dimension 为 5。`LUT/FF/BRAM/DSP/Fmax` 全为 `null`，
`target_measured=false`。这只是 float algorithm 的 operation/storage proxy，不是 RTL、synthesis、
place-and-route、latency 或实板测量。

## 9. 可复现入口

```powershell
python -m cnn_fpga.benchmark.continuous_adaptive_map
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q tests/test_periodic_adaptive_map.py tests/test_continuous_adaptive_map.py
```

机器证据：

- `docs/t3_2_2_continuous_adaptive_map_validation.json`
- `docs/t3_2_2_continuous_adaptive_map_source_data.csv`

