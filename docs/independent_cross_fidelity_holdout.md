# T5.0.2 独立 cross-fidelity holdout

## 1. 结论

本 task 总状态为 `PASS`，含义仅为“至少一个与 calibration 分离的 family 通过，且所有失败 family 被完整
保留”：

- main cross-fidelity family：`FAIL`；
- secondary P-Steane family：`PASS`。

main family 的失败不能被 secondary 的通过抵消，也不能升级 T5.0.1 的 main noise-transfer validity claim。
机器结果为 `docs/t5_0_2_independent_cross_fidelity_holdout.json`，291 行逐点证据为
`docs/t5_0_2_independent_cross_fidelity_holdout_source_data.csv`。

## 2. 数据隔离与冻结规则

T2.3.3 calibration grid 为 `3/5/8/10/12 dB`。在正式点冻结前，只为检查公开 API 和边界行为执行过
`4/11/14 dB` reconnaissance；这些 pilot 点被永久排除。正式点随后冻结为低 squeezing 负对照 `2.5 dB`
与高 squeezing `10.25/11.75 dB`，并使用 fresh seeds
`2026071611/2026071613/2026071617/2026071619`，每点合计 `400,000` 个 effective-model samples。

正式结果出现后不得重选点、重调阈值或删除失败轴。高 squeezing 的冻结门包括：

- noise-transfer/direct-syndrome q-LER gap `<=5e-5`；
- Fock/direct-syndrome q-LER gap `<=5e-4`；
- canonical q/p LER gap `<=1e-6`；
- pooled effective/noise-transfer 最大 axis z-score `<=2.0`；
- validity 为 `localized` 且最小 clipping ratio `>=0.90`。

负对照要求 noise-transfer/direct-syndrome gap `>=0.01`、clipping ratio `<0.5` 且明确标记
`clipping_dominated`。

## 3. Main cross-fidelity 结果

| squeezing | noise/syndrome gap | Fock/syndrome gap | canonical q/p gap | min clipping | max z | 结果 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2.5 dB | 0.0242429 | 2.72174e-7 | 0.0424594 | 0.325684 | 37.0790 | 负边界 PASS；不要求高域一致 |
| 10.25 dB | 3.00789e-5 | 1.50989e-5 | 6.15782e-8 | 0.928374 | **2.293338** | **FAIL** |
| 11.75 dB | 5.19415e-6 | 3.27778e-4 | 5.64306e-10 | 0.974112 | 1.867765 | PASS |

两个高点的三类 deterministic gap、localized 与 clipping gates 都通过，但 `10.25 dB` 的 pooled
effective/noise-transfer 门为 `2.293338 > 2.0`。family gate 要求两个高点同时通过，因此 main family 必须
判 `FAIL`；不得重选 `10.25 dB` 或只报告通过的 `11.75 dB`。

## 4. Secondary P-Steane 解析 holdout

公式来源为 [Chen et al., arXiv:2604.08247v1](https://arxiv.org/pdf/2604.08247) 的 Eqs. 36/37、40、
41、43。独立网格使用 `sigma_A={0.07,0.11,0.19}`、未出现在来源示例中的 data/ancilla variance ratios
`{1.25,2.25,4.75}`、7 个 `b` 和 4 个整数 `m=2a/b`，共 252 点。

- Eq. 40 与由 Eqs. 36/37 系数独立传播的最大误差：`5.55112e-17`；
- Eq. 41 与直接方差乘积的最大相对误差：`4.79556e-16`；
- ME-Steane special case 最大误差：`1.73472e-18`；
- teleportation symmetric special case 最大误差：`6.93889e-18`；
- 所有 `k>1` 网格中 `m=1`（即 `2a=b`）均为唯一 argmin，且乘积为 `sigma_A^4`。

因此 secondary P-Steane family：`PASS`。它只支持 small-noise analytic regression：不进入 sBs 主排名，
不证明物理 squeezing 已实现，也不产生 FPGA、device fidelity 或 hardware claim。

## 5. 非 demo 与 claim 审计

实现不是单点公式展示：主族重放四个 fidelity lanes、3 个正式点、12 个 fresh-seed records；secondary
使用 252 点全网格、两条独立计算路径和 special-case 反查。291 行 Source Data 由 2 个 parent bindings、
3 个主点、12 个 seed、252 个解析点和 22 个 gate rows 组成。T5.0.1 artifact 保持原始预注册快照，不因
本任务完成而回写历史状态。

允许的结论仅为本报告逐族结果。禁止把 task 总 `PASS` 写成 main family 通过，禁止把解析 P-Steane 结果
写成物理实现或硬件实测。
