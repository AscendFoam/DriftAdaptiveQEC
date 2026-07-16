# T2.0.5 sBs 位移故障 syndrome 趋势

**日期：** 2026-07-14  
**实现：** `physics/sbs_displacement_fault.py`  
**机器结果：** `docs/t2_0_5_displacement_fault_trend.json` / `.csv`  
**证据口径：** `protocol_aligned_displacement_fault_trend_not_device_calibrated`

## 1. 一手趋势与归一化

Sivak 2023 Fig. 4(c) 与正文说明：位置位移的 syndrome trace 取决于位移到最近逻辑操作的距离。逻辑位移间隔为 `l_S/2`；`epsilon/l_S=0` 与 `0.5` 分别接近逻辑恒等和逻辑 bit flip，syndrome trace 较短；中点 `0.25` 是 large-distance error，低秩 trickle-down dissipator 需要最多轮次恢复。

本实现使用周期三角距离

```text
d(epsilon) = distance(epsilon/l_S, nearest multiple of 0.5) / 0.25
```

把 `d in [0,1]` 映射为 `Binomial(max_depth=6, p=d)` 的初始 recovery depth。这是透明的项目 effective-model assumption，不是从 Fig. 4(c) 像素拟合出的装置 kernel。

位置位移由 `S_0^z` syndrome 检测，因此 affected constituent 是 Z。按本仓库已冻结的 Kraus 字符 `(Z,X)`、执行顺序 `(X,Z)`，主 correction string 是 `K_eg/K_eg/...`，chronological observation 是 `(X=g,Z=e)`；X 是独立负控。

## 2. 非 demo 接线

- recovery transition 直接读取 T2.0.2 `SBSErrorSpaceInstrument.transition_probabilities`，没有单独硬编码一条展示曲线；
- observed syndrome 逐 constituent 采样 T2.0.3 的 preparation、4×3 readout confusion 和 observation-conditioned reset kernel；
- depth、ideal e-run、observed e-run、restricted recovery time 和 affected/unaffected `P_e(t)` 分开保存；
- main-lobe 9 个幅度各 `4096` shots、`20` full cycles；
- simulation seed `2026071405`，bootstrap seed `2026071406`，95% percentile bootstrap `500` 次；
- `false_e_given_g=0.005`、`e_detection_probability=0.98`、one-step recovery `0.88` 均是显式 sensitivity assumptions，不是 Sivak 装置标定；
- changed-seed、X/Z 变体、坏 readout 和 source-anchor 负测均包含在 direct tests。

## 3. 预注册结果

| `epsilon/l_S` | 最近逻辑操作距离 | mean initial depth | mean observed same-quadrature max e-run | 95% bootstrap CI | restricted mean recovery cycles |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0000 | 0.00 | 0.000 | 0.101 | [0.092, 0.110] | 0.000 |
| 0.0625 | 0.25 | 1.507 | 1.411 | [1.380, 1.445] | 1.714 |
| 0.1250 | 0.50 | 2.995 | 2.626 | [2.593, 2.663] | 3.423 |
| 0.1875 | 0.75 | 4.520 | 3.835 | [3.795, 3.872] | 5.144 |
| 0.2500 | 1.00 | 6.000 | 4.883 | [4.846, 4.919] | 6.805 |
| 0.3125 | 0.75 | 4.501 | 3.826 | [3.791, 3.860] | 5.119 |
| 0.3750 | 0.50 | 3.006 | 2.635 | [2.598, 2.668] | 3.420 |
| 0.4375 | 0.25 | 1.504 | 1.417 | [1.390, 1.443] | 1.704 |
| 0.5000 | 0.00 | 0.000 | 0.104 | [0.096, 0.113] | 0.000 |

所有幅度在 20-cycle horizon 内 recovery fraction 为 `1.0`；这只是当前 assumption 下没有 censored shot，不是装置保证。

## 4. 容差门与结果

| Gate | 容差 | 实测 | 结论 |
| --- | ---: | ---: | --- |
| peak location | `abs(peak-0.25) <= 0.0625` | 0.2500 | PASS |
| midpoint/endpoints run CI separation | `>= 2.0` | 4.7327 | PASS |
| left Spearman | `>= 0.95` | 1.0000 | PASS |
| right Spearman | `<= -0.95` | -1.0000 | PASS |
| mirror max run difference | `<= 0.30` | 0.0095 | PASS |
| endpoint mean initial depth | `<= 0.05` | 0.0000 | PASS |
| midpoint depth error from 6 | `<= 0.05` | 0.0000 | PASS |
| midpoint early-minus-late affected `P_e` | `>= 0.25` | 0.8606 | PASS |
| unaffected-axis max `P_e` | `<= 0.06` | 0.0273 | PASS |
| midpoint recovered fraction | `>= 0.98` | 1.0000 | PASS |

JSON 保存每个 gate 的 `check_id/criterion/limit/observed/detail`。`require_pass()` 会列出所有失败 ID，而不是只返回一个模糊布尔值。将 false-e 与 e-detection 同时改为 0.5 时，负控等 gate 会按名称失败，证明诊断没有被写死成 PASS。

## 5. 证据边界

本任务复现的是文献一致的非单调方向、同象限 syndrome string 和低秩多轮恢复语义。它没有 digitize Fig. 4(c)，没有拟合实验 `P_e(t,epsilon)`，也没有模拟 Fock-space coherent displacement、ECD pulse、cavity-transmon Hamiltonian 或真实 readout/reset calibration。可写 `qualitative protocol-aligned trend reproduced`；不可写 `quantitative experimental reproduction`、`device-calibrated digital twin` 或真实硬件因果结果。
