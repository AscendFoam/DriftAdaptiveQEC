# T6.18.2 official structured-lattice CPD 复现

- verdict：`PASS_OFFICIAL_CPD_SMALL_DISTANCE_THRESHOLD_REPRODUCTION_WITH_UPSTREAM_CAVEAT`
- official commit：`01f9bf1f6970b3e229b43aac9da3325c75518db8`
- independent paired trials：1,728,000
- gates / mutations：16/16 / 17/17

## 结果

| 方法 | d3/d5 crossing | d5/d7 crossing | mean crossing [bootstrap 95% CI] | anchor |
| --- | ---: | ---: | ---: | ---: |
| `cpd` | 0.601232 | 0.597364 | 0.599298 [0.596451, 0.602584] | 0.602 |
| `analog_mwpm` | 0.602435 | 0.598056 | 0.600245 [0.598032, 0.602991] | 0.599 |

本次冻结网格中，CPD 在 27/27 个 d×σ cell 的 LER 低于 analog-MWPM；平均 CPD−analog LER 为 -0.011621。这是同一 stationary surface-GKP 小距离 family 的配对结果，不是相对所有解码器的主排名。

官方 Fig. 5 聚合数据重算逐位得到 CPD `0.6024563484 ± 0.0003776410` 与 analog-MWPM `0.5995937637 ± 0.0004433259`。该结果来自作者提供的 10^7 samples/point JLD2，不是本项目独立 Monte Carlo。

## 正确性与实现边界

- 1–4 维 certified brute-force CVP：0/312 mismatch。
- d=3 official fast CPD vs generic exact CVP：0/64 mismatch。
- final-list vs canonical lattice logical coordinates：0/384 mismatch。
- analog comparator 使用 Noh–Chamberland Eq. (11) 的条件逻辑错误概率及 Appendix B 的 `-log2(p)`；它是 source-transcribed adapter，不是官方仓库原生函数。
- 标准上游测试保留 2004/2005 的无 seed 随机失败；固定 seed 重放为 2005/2005。官方源码未修改。

## 不能声称的内容

d=3/5/7、64,000 trials/point 只支持 ±0.02 粗阈值复现，不能替代论文 d=15–29、10^7 samples/point 的三位小数结论。三尺寸 runtime exponent 只是经验诊断；论文 `d^3.020` 保持 literature-only。没有 FPGA 真板结果，也没有把 Phase 6B NO-GO 改写为通过。

## Artifacts

- `docs/t6_18_2_official_structured_cpd_source_data.csv`
- `docs/t6_18_2_official_structured_cpd_reproduction.json`
- `scripts/run_lattice_algorithms_reproduction.jl`
- `cnn_fpga/benchmark/official_structured_cpd_reproduction.py`
