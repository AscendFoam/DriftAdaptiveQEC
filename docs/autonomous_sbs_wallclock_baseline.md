# T3.2.8 autonomous sBs 物理时间 baseline

## 1. 结论先行

本任务得到一个必须保留单位限定的负结果。在同一组 nominal sBs controls、同一有限 Fock
cutoff 和同一文献噪声参数下，autonomous sBs 因 7 us full cycle 比 measurement-feedback
的 10 us 更短，按“各自协议 cycle 数”报告时 6/6 条件都显得更好；换成共同 700 us 物理时间后，
其 projected-logical-Z area-equivalent lifetime 却在 6/6 条件都更差：比值范围从
`0.805901` 到 `0.942271`。因此不得把文献的 `0.7×` cycle duration 直接乘到 lifetime，
也不得只报告 per-cycle 数字。

这不是 autonomous pulse optimization，也不是 Puviani Fig. 3(b) 数值复现。它是由项目现有
finite-cutoff joint cavity--two-level-ancilla 模型给出的、protocol-native、固定控制参数的
wall-clock baseline。

## 2. 文献时间与协议 contract

本地 Puviani 正文/补充材料给出 measurement-feedback half cycle 为 5 us、full cycle 为
10 us，并在 Table S1 列出各物理阶段；autonomous 路径省略 measurement，但仍在第 4 层后
reset ancilla，full cycle 为 standard cycle 的 `0.7`。实现把两条路径分别展开为：

| 路径 | half cycle | full cycle | 每 half-cycle measurement | 每 half-cycle reset |
| --- | ---: | ---: | ---: | ---: |
| measurement-feedback | 5.0 us | 10.0 us | 1 | 1 |
| autonomous | 3.5 us | 7.0 us | 0 | 1 |

两条路径共享 `enter/layer1/layer2/layer3/layer4/virtual-rotation-and-idle =
0.1/0.5/0.7/0.3/0.1/1.0 us`。measurement-feedback 的 measurement+reset 为 2.3 us；
autonomous 只保留 0.8 us reset。所有时间均标记为 literature simulation timing，
`target_hardware_measured=false`。

## 3. 不是 lifetime 缩放的数值实现

`physics/autonomous_sbs.py` 复用 T2.3.4 已验证的显式 cavity--ancilla gates 与 idle CPTP maps，
但分别执行两条真实离散演化：

- measurement-feedback 路径显式完成 ancilla outcome 的非选择性 measurement，再 trace/reset；
- autonomous 路径不产生 measurement event，但仍 trace/reset ancilla；
- 两条路径各自按原生 half-cycle duration 施加 cavity/ancilla decoherence；
- 在共同 700 us horizon 上分别执行 70 与 100 个 full cycles，而不是把某条 lifetime 乘以 0.7。

两个独立方法审计封住了实现捷径：四个 `gg/ge/eg/ee` 分支概率和为
`1.0000000000000002`，显式分支枚举与非选择性 channel 的最大 density difference 为
`1.42e-16`；把 decoherence 关掉后，两条 duration 路径的同 cycle 最大 density difference
为 `5.38e-15`。这证明差异来自 wall-clock noise exposure/event structure，而不是两套隐藏 gate。

## 4. 扫描与结果

生产矩阵覆盖 cutoff `12/16` 与 Puviani Table S5 的 high/medium/low 三组
`(cavity lifetime, ancilla T1, ancilla T2)`，分别为 `(245,50,60)`、`(490,100,120)`、
`(610,280,238)` us。共 6 lanes、1,020 full cycles，采用 deterministic nonselective
density evolution，无 Monte Carlo CI。

| cutoff | noise | measurement lifetime (us) | autonomous lifetime (us) | autonomous / measurement (us) | autonomous / measurement (protocol cycles) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 12 | high | 59.373650 | 55.946057 | 0.942271 | 1.346101 |
| 12 | medium | 67.621410 | 61.207151 | 0.905145 | 1.293064 |
| 12 | low | 69.338785 | 62.307519 | 0.898595 | 1.283708 |
| 16 | high | 102.897911 | 88.066965 | 0.855867 | 1.222668 |
| 16 | medium | 121.199183 | 98.587273 | 0.813432 | 1.162045 |
| 16 | low | 126.049949 | 101.583758 | 0.805901 | 1.151287 |

表中 lifetime 是 projected logical-Z signal 在 `[0,700 us]` 上的 normalized signed AUC
对应面积等效寿命。cutoff 12 的三个 autonomous 末端 logical-Z 单点反而比 measurement
高 `0.00205--0.00564`，但全时域 lifetime 仍更差；自动化测试专门锁定这个冲突，禁止用
单个 700 us endpoint 替代 lifetime。

## 5. 原始事件账本

每个 lane 在共同 700 us 的账本完全相同：

| 路径 | full cycles | measurements | resets | active gate applications |
| --- | ---: | ---: | ---: | ---: |
| measurement-feedback | 70 | 140 | 140 | 1,260 |
| autonomous | 100 | 0 | 200 | 1,800 |

因此 autonomous 在共同时间省去 140 次 measurement，但多 60 次 reset 与 540 次 active
gate application；reset rate 和 active-gate rate 都是 measurement-feedback 的 `10/7`。
本任务保留这些原始计数，不把 measurement/reset/gate 主观压成一个没有校准权重的成本标量。

## 6. 证据、复现与 claim 边界

- runner：`python -m cnn_fpga.benchmark.autonomous_sbs_wallclock_baseline`；
- machine artifact：`docs/t3_2_8_autonomous_sbs_wallclock_validation.json`，17/17 gates；
- Source Data：`docs/t3_2_8_autonomous_sbs_wallclock_source_data.csv`，4,362 rows；
- direct/artifact tests：`tests/test_autonomous_sbs.py` 与
  `tests/test_autonomous_sbs_wallclock_baseline.py`。

允许的论文表述仅限：finite-cutoff nominal-control model 在 protocol-cycle 与共同文献
wall-clock 两种口径下的 signed comparison。禁止写成 trained autonomous optimum、论文
Fig. 3(b) reproduction、multilevel leakage/pulse dynamics、目标板或装置实测 timing。尤其不能
把当前 negative ranking 外推为 autonomous QEC 的一般劣势；它同时受固定 controls、finite
cutoff、two-level ancilla 和当前模型的影响。

