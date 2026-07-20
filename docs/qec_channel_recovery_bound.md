# T5.3.5：QEC-matrix/Petz channel-recovery bound

## 结论

本任务建立了一个**有限截断、单周期纯损耗、允许任意 terminal CPTP recovery** 的 encoding--noise
性能界。它回答“当前有限能 GKP 编码在这条噪声通道后理论上还能恢复到什么程度”，不回答“当前 sBs
控制器或 FPGA decoder 已经做到什么程度”。机器产物 21/21 gates 通过，Source Data 共 119 行。

small-cutoff 的 15 条真实 GKP lanes 同时通过 Petz 解析双边界与 recovery-Choi SDP primal/dual 证书；
Petz scan 又扩到 cutoff 48，并对三档 `Delta` 做能量敏感度。actual sBs 只得到时序不匹配的诊断差值；
T4.4.4 teacher/student 因 horizon 和 metric 不同，gap 保持 `null/INCOMPARABLE`。

## 数学对象

令 `V:C^2 -> C^N` 为 orthonormal finite-cutoff GKP isometry，`N` 为 `10 us` exact finite-Fock
cavity pure-loss channel，`A_k=N_k V`。QEC matrix 采用 Zheng 等人的逻辑指标优先约定：

```text
M[mu,l;nu,k] = <mu| A_l^dagger A_k |nu>.
```

transpose/Petz recovery 的 channel fidelity 为

```text
F_Petz = ||Tr_L sqrt(M)||_F^2 / d_L^2,
F_Petz <= F_opt <= (1 + F_Petz)/2.
```

这里的 `F_opt` 对所有 physical-output→logical CPTP maps 优化。参考来源是
[The Near-optimal Performance of Quantum Error Correction Codes](https://arxiv.org/abs/2401.02022)
的 Eq. (4)/(5) 与 supplement recovery SDP；本文实现还用显式 Petz Kraus
`A_k^dagger [A(I)]^{-1/2}` 独立复核 QEC-matrix fidelity。

## small-cutoff SDP 证书

recovery Choi 使用 output-fast ordering。primal 最大化 `Tr(C J_R)`，约束 `J_R>=0` 且
`Tr_output J_R=I_physical`；dual 最小化 `Tr(Y)`，约束 `kron(Y,I_L)-C>=0`。

不直接相信 solver 打印的目标值：

1. raw primal 先 Hermitian/PSD 投影，再用 partial-trace inverse square root 做 CPTP 归一化；所得目标是
   可行下界。
2. raw dual 按 slack 最小特征值加 identity shift，直到留出 `1e-10` 正 margin；所得 trace 是可行上界。
3. 只有 repaired primal/dual 和 Petz theorem 两个区间相交才通过。

cutoff 4/6/8/10/12×high/medium/low 共 15 条 lane 全部通过。9 条 raw CLARABEL solve 至少一侧为
`optimal_inaccurate`，但没有隐藏：修复后最大 certificate width 为 `1.484154e-7`，最大 raw duality-gap
也在门限内；Petz fidelity 到 SDP feasible optimum 的最大差为 `0.001788472`。

## cutoff 与能量扩展

固定 `Delta=0.34` 时：

| cutoff | mean code photon number | high-noise `F_Petz` | medium | low |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 2.968912 | 0.999317991 | 0.999811366 | 0.999868710 |
| 40 | 3.837866 | 0.999832713 | 0.999965026 | 0.999977106 |
| 48 | 3.844109 | 0.999833198 | 0.999965118 | 0.999977153 |

cutoff 48、high noise 下，`Delta=0.44/0.34/0.28` 的 mean photon number 为
`2.137270/3.844109/5.865476`，对应 `F_Petz=0.997879436/0.999833198/0.999986929`。
这证明代码能够执行更大 cutoff/更高注册能量的有限维 scan，不证明无限维极限。terminal cutoff Petz
最大差为 `4.827748e-6`；noise-output support inverse square root 的 TP residual 最大为
`1.181287e-4`，而 QEC 公式与 direct Petz fidelity 最大差仍仅 `1.520695e-10`。两种数值量均原样保存。

## actual sBs 诊断，不是严格 controller gap

T5.3.2 的 nominal sBs 在两个 half-cycle 中交错 gate、reset、cavity loss 与 ancilla noise；本 bound 则是
先让 encoded cavity 经 `10 us` pure loss，再允许一次任意 terminal recovery。二者不属于同一 scheduling
feasible set。

为避免把 CPTNI leakage 免费丢弃，actual sBs 先追加固定 terminal decode：code-space 内做 `V^dagger rho V`，
leakage 以 maximally mixed logical state 补全。这会给 `F_e` 增加 `mean_leakage/d_L^2`。例如 cutoff 40：

| noise | completed actual sBs `F_e` | Petz/theorem interval | bound-minus-sBs diagnostic |
| --- | ---: | ---: | ---: |
| high | 0.693953 | [0.999833, 0.999916] | [0.305880, 0.305964] |
| medium | 0.719020 | [0.999965, 0.999983] | [0.280945, 0.280962] |
| low | 0.724502 | [0.999977, 0.999989] | [0.275475, 0.275487] |

这些数值只能写成 `SCHEDULE_MISMATCHED_DIAGNOSTIC_ONLY`。它们提示 terminal arbitrary-recovery 空间与
当前 nominal sBs 过程之间仍有大差距，但不能证明 sBs 是 SDP 的可行点、Petz 可部署、或两者严格排序。

## teacher/student gap

T4.4.4 的 teacher/student 使用 cutoff 12/16、10-cycle two-level history-conditioned trajectories，指标是
selection score、fidelity lifetime 与 logical-Z lifetime；它没有与本任务相同的 one-cycle six-state Choi。
因此四个 role/cutoff 单元均报告：

```text
recovery_bound_gap = null
status = INCOMPARABLE
```

禁止把 lifetime 与 one-cycle `F_e` 直接相减。

## 产物与复现

- `physics/channel_recovery_bound.py`
- `cnn_fpga/benchmark/qec_channel_recovery_bound.py`
- `docs/t5_3_5_qec_channel_recovery_bound.json`
- `docs/t5_3_5_qec_channel_recovery_bound_source_data.csv`
- `tests/test_channel_recovery_bound.py`
- `tests/test_qec_channel_recovery_bound.py`

```powershell
$env:PYTHONPATH='.'
python -m cnn_fpga.benchmark.qec_channel_recovery_bound
python -m pytest -q tests/test_channel_recovery_bound.py tests/test_qec_channel_recovery_bound.py
```

## Claim 边界

允许：finite-cutoff pure-loss encoding--noise Petz/theorem bound、small-cutoff repaired primal/dual SDP
certificate、cutoff/energy sensitivity，以及明确标注时序不匹配的 sBs diagnostic。

禁止：Petz decoder、可部署 controller、actual sBs 的 certified optimality gap、teacher/student 数值 gap、
large-cutoff SDP optimum、无限 cutoff/energy convergence、physical-memory LER、QPU、RTL、板卡或实验结果。
