# Paper-parameter registry

**Task：** T1.4.4  
**状态：** Frozen contract  
**机器真源：** `docs/paper_parameter_registry.json`

## 1. 目的与判定边界

本 registry 把论文里的协议、时序、读出类别、误分类、reset、漂移、leakage、
squeezing/noise ratio 和逻辑指标，与当前仓库的工程默认值严格分开。它不是“可直接抄到
config 的参数表”，而是一个带来源、归一化、迁移边界和校准门的证据表。

四类状态只有以下含义：

| classification | 含义 | 可否直接进入项目默认值 |
| --- | --- | --- |
| `literature_fact` | 本地一手论文正文/补充材料直接支持；仍只属于原设备、原协议或原模型 | 否 |
| `modeling_assumption` | 当前仓库配置/代码采用的值，不是论文事实，也不是实板测量 | 仅可作为现有模型默认值 |
| `secondary_reference` | 仓库阅读卡或综述级证据；尚未在本 task 回到一手正文复核 | 否 |
| `pending_calibration` | 当前无可信可迁移数值，机器值必须保持 `null` | 通过条目 gate 后才可填 |

所有 51 个条目的机器记录都包含 `source_path + line_start/line_end +
expected_fragment`（待校准条目除外）、`normalization`、`allowed_use`、
`forbidden_transfer` 和 `calibration_gate`。因此“有论文数字”不等于“可迁移数字”。

## 2. 来源与版本去重

| Source ID | 记录 | 证据层 | 版本处理 |
| --- | --- | --- | --- |
| `SRC-CAMPAGNE-2020` | Campagne-Ibarcq et al., Nature 2020, DOI `10.1038/s41586-020-2603-3` | primary local | 正式记录 |
| `SRC-SIVAK-2023` | Sivak et al., Nature 2023, DOI `10.1038/s41586-023-05782-6` | primary local | 正式记录优先于 arXiv 2211.09116 |
| `SRC-PUVIANI-2025` | Puviani et al., PRL 2025, DOI `10.1103/physrevlett.134.020601` | primary local | 正式记录 |
| `SRC-GKP-2001` | Gottesman--Kitaev--Preskill, PRL 2001, DOI `10.1103/physrevlett.87.127901` | primary local | OCR 只取结构性事实 |
| `SRC-WAN-2020-CARD` | Wan et al., PR Research 2020, DOI `10.1103/physrevresearch.2.043280` | secondary local | 强数值须回一手正文 |
| `SRC-LACHANCE-2024-CARD` | Lachance-Quirion et al., PRL 2024, DOI `10.1103/physrevlett.132.150607` | secondary local | 正式记录优先于 arXiv 2310.11400；本 task 只用阅读卡 |
| `SRC-PROJECT-HIL-CONFIG` | `cnn_fpga/config/hardware_hil.yaml` | project local | modeling assumption |
| `SRC-PROJECT-CONSTANTS` | `physics/quadrature_conventions.py` | project local | chart-qualified normalization contract |

DOI-first 去重后，同一工作只保留一个 formal record；preprint key 只作为 provenance，
不重复进入论文计数。对一手 OCR 文本，registry 只选可由行内文本复核的数值；图中但正文未
展开的完整 confusion matrix 不做人工猜读。

## 3. 最关键的不可混写口径

### 3.1 Cycle 与时间

- `PF-S01`：Sivak 的 constituent/even-or-odd quadrature step 是 `4.924 us`；完整
  `X+Z` composite QEC cycle 是 `9.848 us`。logical error per cycle 绑定完整 cycle。
- `PF-S10`：同一最高-gain SBS 在 prose 为 `1546 ns`，Table S3 各层求和为
  `1548 ns`；保留 `2 ns` 源内差异，正式引用前回原 PDF，不擅自修正。
- `PF-P02`：Puviani 数值模型采用 `10 us` full cycle、`5 us` half-cycle，且
  measurement+numerical reset 占每 half-cycle 的 `2.3 us`。这不是实验实测。
- `MA-H01`：项目 `t_fast_us=5.0` 只是 runtime 配置；既不等于 Puviani half-cycle，
  也不证明低价 FPGA 已达到 5 us 闭环。
- `PC-T01`：低价板 I/O+compute+commit+ack 的 p50/p95/p99/max 仍为 `null`。

### 3.2 读出类别与误分类

- `PF-C04`：Campagne QEC round 是 transmon `sigma_y` 的 `|+y>/|-y>` syndrome 类；
  logical Pauli readout 的 `±1` 是另一层语义。
- `PF-S03`：Sivak 用两个 IQ threshold 编码 `g/e/f`；这是原微波读出链的三类状态。
- `PF-P03/PF-P04`：Puviani 模型只有理想 projective `g/e`，未建 readout
  classification error；`null` 不表示真实误分类率为零。
- `PF-C05` 只保存 `>99%` readout lower bound 和 `99.5%` Rabi contrast；
  `PF-S04` 只保存 `F_g=0.9997`、`F_e=0.9914`。两篇都没有可从当前文本完整复核并迁移
  到项目的 class-conditional matrix。
- `MA-M01` 的 `ancilla_error_rate=0.01` 与 `MA-R01` 的
  `measurement_efficiency=0.95` 是 effective assumptions，不是上述 fidelity 的替代。

### 3.3 Reset 与 leakage

- `PF-C06/PF-C08`：Campagne 每轮以 measurement-conditioned `pi/2` reset；进入
  `f` 后 g/e control 无效，原设备报告约 `(3 ms)^-1` 的 higher-level jump rate。
- `PF-S05/PF-S07`：Sivak prose reset subroutine 为 `2332 ns`，Table block 若含
  entry/exit 为 `2380 ns`；高于 `f` 的未覆盖 leakage 有 `17.2` constituent-cycle
  衰减长度，rate 只能归属原设备和明确 cycle 口径。
- `PF-P05/PF-P06`：Puviani 直接数值 reset 到 `g`，并省略 leakage/SPAM；这是模型
  理想化，不是硬件能力。
- `PC-X01/PC-L01`：项目 reset duration/failure、leakage taxonomy/rate/memory 均保持
  `null`，不能拿任一论文数值填充。

### 3.4 Squeezing/noise 与逻辑指标

- `PF-C09`：Campagne steady-state peak squeezing `7.4--9.5 dB`、平均光子数
  `8.6--10.2` 是 master-equation inferred range，须绑定其 vacuum convention。
- `PF-P07`：Puviani 通用设置 `Delta=0.34, nbar≈5` 与 headline best-agent 的
  `Delta=0.2` scope 不同，不能合并为一个全局值。
- `PF-P09`：Puviani low/medium/high 三组 `Ts/T1/T2` 是以 `10 us` model cycle
  换算的 numerical-study assumptions，不是本文测量或项目校准值。
- `PF-G01` 只冻结“Gaussian peaks + Gaussian envelope”的 finite-squeezing 结构；
  不从 OCR 公式抽未复核 normalization。
- `MA-N01/MA-N02/MA-U01` 的 `sigma_measurement=0.03`、`sigma_p/sigma_q=0.55`、
  decoder `LATTICE_CONST=sqrt(2*pi)` 是仓库约定。T-RISK-20260714-01 已冻结
  canonical logical cell `sqrt(pi)`、decoder classical scale `sqrt(2)`、paper displacement
  stabilizer amplitude `l_S=sqrt(2*pi)` 及 dB/variance 解析换算；device envelope/`nbar`
  仍不能直接互换。
- `PF-C10`、`PF-S08` 是原实验 lifetime/gain；`PF-P08` 是 best-of-20 的数值模拟
  lifetime。项目 `PC-G01` 仍没有同系统 passive comparator 和 physical break-even。

## 4. 完整条目索引

下面的 ID 与 JSON 一一对应；测试会检查 Markdown 与 JSON 不允许漏项或多项。

### 4.1 一手文献事实（30）

| ID | 类别 | 内容摘要 |
| --- | --- | --- |
| <!-- registry-id: PF-C01 --> `PF-C01` | protocol | square four-round sharpen/trim block |
| <!-- registry-id: PF-C02 --> `PF-C02` | timing | conditional displacement `1.1 us` |
| <!-- registry-id: PF-C03 --> `PF-C03` | timing | transmon readout `700 ns` |
| <!-- registry-id: PF-C04 --> `PF-C04` | readout | `sigma_y`, `+y/-y` syndrome classes |
| <!-- registry-id: PF-C05 --> `PF-C05` | misclassification | readout lower bounds，matrix absent |
| <!-- registry-id: PF-C06 --> `PF-C06` | reset | conditional `pi/2` reset，duration/failure absent |
| <!-- registry-id: PF-C07 --> `PF-C07` | drift | original DAC relative drift sensitivity `1e-3` |
| <!-- registry-id: PF-C08 --> `PF-C08` | leakage | `f` leakage mechanism and original rate scale |
| <!-- registry-id: PF-C09 --> `PF-C09` | squeezing/noise | `7.4--9.5 dB`, `8.6--10.2 photons` |
| <!-- registry-id: PF-C10 --> `PF-C10` | logical | square QEC-on `T_X/T_Y/T_Z` |
| <!-- registry-id: PF-S01 --> `PF-S01` | protocol | `4.924 us` constituent / `9.848 us` full X+Z |
| <!-- registry-id: PF-S02 --> `PF-S02` | timing | readout/acquisition/FPGA-DSP/bit-distribution breakdown |
| <!-- registry-id: PF-S03 --> `PF-S03` | readout | IQ-threshold `g/e/f` classes |
| <!-- registry-id: PF-S04 --> `PF-S04` | misclassification | partial `F_g/F_e`, full matrix absent |
| <!-- registry-id: PF-S05 --> `PF-S05` | reset | `2332 ns` prose / `2380 ns` table-block scope |
| <!-- registry-id: PF-S06 --> `PF-S06` | drift | original system retraining every `1--2 weeks` |
| <!-- registry-id: PF-S07 --> `PF-S07` | leakage | duration/rate with explicit original cycle scope |
| <!-- registry-id: PF-S08 --> `PF-S08` | logical | peak lifetime, gain and error per full cycle |
| <!-- registry-id: PF-S09 --> `PF-S09` | squeezing/noise | original device coherence window |
| <!-- registry-id: PF-S10 --> `PF-S10` | timing | SBS `1546/1548 ns` source-internal discrepancy |
| <!-- registry-id: PF-P01 --> `PF-P01` | protocol | model-based measurement sBs half-cycle |
| <!-- registry-id: PF-P02 --> `PF-P02` | timing | `10 us` full / `5 us` half model cycle |
| <!-- registry-id: PF-P03 --> `PF-P03` | readout | ideal projective `g/e` observation |
| <!-- registry-id: PF-P04 --> `PF-P04` | misclassification | classification-error model absent |
| <!-- registry-id: PF-P05 --> `PF-P05` | reset | numerical reset，not gates/pulses |
| <!-- registry-id: PF-P06 --> `PF-P06` | leakage | leakage/SPAM omitted |
| <!-- registry-id: PF-P07 --> `PF-P07` | squeezing/noise | scoped `Delta=0.34` and `Delta=0.2` |
| <!-- registry-id: PF-P08 --> `PF-P08` | logical | best-of-20 numerical lifetime result |
| <!-- registry-id: PF-P09 --> `PF-P09` | squeezing/noise | scoped low/medium/high model noise levels |
| <!-- registry-id: PF-G01 --> `PF-G01` | squeezing/noise | finite squeezing structural fact |

### 4.2 Secondary references（3）

| ID | 类别 | 内容摘要 / 升级门 |
| --- | --- | --- |
| <!-- registry-id: SR-W01 --> `SR-W01` | protocol | Wan multi-round Bayesian memory；回正式正文后才可升格 |
| <!-- registry-id: SR-W02 --> `SR-W02` | squeezing/noise | reading-card `Delta≈0.22/nbar≈10`；须核验 Figure 3 与 convention |
| <!-- registry-id: SR-L01 --> `SR-L01` | protocol | Lachance autonomous sBs 邻域；须核验 reset error/lifetime |

### 4.3 当前 modeling assumptions（9）

| ID | 类别 | 当前值 | 明确不是 |
| --- | --- | --- | --- |
| <!-- registry-id: MA-H01 --> `MA-H01` | timing | `t_fast_us=5.0` | 实板 latency / paper cycle |
| <!-- registry-id: MA-H02 --> `MA-H02` | timing | `window_size=2048` | Bayesian history horizon |
| <!-- registry-id: MA-H03 --> `MA-H03` | timing | `t_slow_update_ms=20.0` | quantum feedback latency |
| <!-- registry-id: MA-N01 --> `MA-N01` | squeezing/noise | `sigma_measurement=0.03` | squeezing dB / Delta |
| <!-- registry-id: MA-N02 --> `MA-N02` | squeezing/noise | `sigma_ratio_p=0.55` | 论文实测 noise ratio |
| <!-- registry-id: MA-M01 --> `MA-M01` | misclassification | `ancilla_error_rate=0.01` | g/e/f confusion matrix |
| <!-- registry-id: MA-R01 --> `MA-R01` | readout | `measurement_efficiency=0.95` | assignment fidelity |
| <!-- registry-id: MA-P01 --> `MA-P01` | protocol | effective `delta=0.3` | Campagne shift `delta≈0.2` |
| <!-- registry-id: MA-U01 --> `MA-U01` | squeezing/noise | decoder logical cell `sqrt(2*pi)` | canonical operator pair / paper `l_S` 的同一语义 |

### 4.4 待校准（9）

| ID | 类别 | 当前值 | Gate |
| --- | --- | --- | --- |
| <!-- registry-id: PC-P01 --> `PC-P01` | protocol | `null` | T2 protocol schema + trajectory tests + normalization map |
| <!-- registry-id: PC-T01 --> `PC-T01` | timing | `null` | T6 实板同口径 latency provenance |
| <!-- registry-id: PC-R01 --> `PC-R01` | readout | `null` | T2 observed-event schema + replay provenance |
| <!-- registry-id: PC-M01 --> `PC-M01` | misclassification | `null` | labeled counts + frozen thresholds + CI |
| <!-- registry-id: PC-X01 --> `PC-X01` | reset | `null` | reset FSM + coverage tests + data source |
| <!-- registry-id: PC-D01 --> `PC-D01` | drift | `null` | timestamped trace fit + held-out validation |
| <!-- registry-id: PC-L01 --> `PC-L01` | leakage | `null` | state machine + primary/authorized calibration |
| <!-- registry-id: PC-N01 --> `PC-N01` | squeezing/noise | `null` | 坐标/解析与 source-scenario coherent short-trajectory 子门已通过；剩余 device state-moment/envelope/`nbar` 校准 |
| <!-- registry-id: PC-G01 --> `PC-G01` | logical | `null` | same-system passive comparator + cycle + CI |

## 5. 迁移与 claim 规则

1. Formal DOI record 优先，preprint 不重复计数；metadata refresh 不能静默改变 canonical key。
2. 原论文设备数值只能说明 source system；不能填入 `hardware_hil.yaml`。
3. fidelity、contrast、assignment error、SPAM、reset failure 和 leakage rate 是不同量；不得互相代替。
4. `cycle` 必须带 `constituent`、`half-cycle` 或 `full X+Z` 修饰。
5. `Delta`、peak `sigma`、envelope、vacuum variance、dB、`nbar` 和仓库
   `LATTICE_CONST` 必须经显式 normalization map。
6. `PF-P08` 只能写成 model-based best-agent numerical result；不能写为 experimental、
   unbiased single-training 或 deployed-controller result。
7. 当前项目没有真实 quantum readout、reset、leakage、physical lifetime 或 low-cost-board
   closed-loop measurement；对应项保持 `pending_calibration`。

## 6. 非 demo 审计结论

本 task 没有用“抄几篇论文数字”替代 registry contract：每条事实均有机器锚点和禁迁移语义；
三篇核心论文中的实验事实、数值模型理想化和 source-internal scope differences 分开保存；
secondary cards 不能升级强 claim；项目默认值由测试直接与当前 YAML/constant 比对；九类真实缺口
全部以 `null + gate` 关闭 fail-open 路径。后续任务若改变配置、单位或证据层，必须同步更新 JSON、
Markdown ID 索引和 tests。
