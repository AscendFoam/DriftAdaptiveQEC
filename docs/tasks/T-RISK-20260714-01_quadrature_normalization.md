# T-RISK-20260714-01 quadrature normalization 与 Fourier-p 修复

- **Task ID：** T-RISK-20260714-01
- **来源风险：** R-N041
- **日期：** 2026-07-14
- **状态：** Done

## 输入与通过标准

输入包括 T2.3.3 的旧 Fourier-p negative audit、`physics/constants.py`、
`finite_energy_gkp.py`、Fock preparation/cross-fidelity 路径、PC-N01，以及 GKP、
Campagne-Ibarcq 2020、Sivak 2023 一手 convention。

通过标准：

1. 分开 canonical、decoder-standardized、displacement-amplitude 与 symplectic chart；
2. commutator、reciprocal lattice、wavefunction Jacobian、covariance/dB map 可解析核验；
3. peak/moment/alias 与 vector/covariance/Fourier roundtrip 有独立数值测试；
4. 修复 10/12 dB Fock q/p audit，并保留旧错误路径作为负证据；
5. 若只修 domain spacing、仍漏 width/envelope/Jacobian，则判定失败；
6. device calibration 与 coherent joint-axis process fidelity 保持 fail closed。

## 执行方案与实际完成

### 1. 一手 convention 核验

按 `nature-academic-search` 的 citation-verification + minimal multi-source-search 流程，
用本地一手全文与官方 arXiv 核对：

- canonical `[x,p]=i` 的 logical spacing `sqrt(pi)` 与 stabilizer `2*sqrt(pi)`；
- Sivak complex displacement amplitude 的 `l_S=sqrt(2*pi)` 与 logical `l_S/2`；
- `x=(a+a†)/sqrt(2)`、`p=i(a†-a)/sqrt(2)` 与 Fourier rotation；
- square-grid matrix `det M=1`。

学术 MCP 当前不可用，依 skill 降级规则使用本地 primary source + 官方 arXiv，没有用二手
网页代替公式证据。

### 2. 根因定位

旧实现把数值相同但语义不同的两个 `sqrt(2*pi)` 混在一起：decoder logical cell 与
paper displacement stabilizer amplitude。更深一层的问题是 damped-projector 仅缩 comb
centers，没有同步 peak width、envelope inverse width、noise variance 和 wavefunction
Jacobian；因此旧 audit 不是单个常数错误。

### 3. 代码修复

- 新增 `physics/quadrature_conventions.py`：四 chart、det/commutator、phase vector、
  covariance、axis sigma、wavefunction dilation、logical spacing 与 15-gate validator；
- `physics/finite_energy_gkp.py`：decoder state 变为 canonical Mehler state 的完整
  `sqrt(2)` dilation；
- `physics/fock_density_model.py`：标准 preparation 只接受注册的 decoder→canonical bridge；
  generic arbitrary scale 保留在低层 API；cutoff gate 加强到 `>0.99999`；
- `physics/cross_fidelity_validation.py`：canonical q/p folding、axis-resolved Pauli metrics、
  legacy ambiguous negative path、15 production gates；
- `physics/noise_transfer_surrogate.py` 与 `finite_squeezing_noise.py`：dB/peak/vacuum
  variance chart-qualified；decoder vacuum variance 修正为 1；
- `physics/constants.py`、`physics/__init__.py`、protocol hierarchy 与 PC-N01 同步。

### 4. 文档与历史证据回灌

更新 T1.2.1、T1.2.3、T2.2.1、T2.3.1、T2.3.2、T2.3.3、T2.3.8 的 task/docs 与机器
JSON。旧数值不静默消失，均以 T-RISK correction note 记录行为变化和 claim 影响。

## 产物路径

- `physics/quadrature_conventions.py`
- `tests/test_quadrature_conventions.py`
- `docs/quadrature_normalization_contract.md`
- `docs/t_risk_20260714_01_quadrature_validation.json`
- `docs/t2_3_1_fock_density_validation.json`
- `docs/t2_3_2_fock_sbs_cycle_validation.json`
- `docs/t2_3_3_cross_fidelity_validation.json`
- `docs/t2_3_8_noise_transfer_validation.json`

## 验证方式和结果

- quadrature direct：`32 passed`；
- 相关 finite-energy/finite-squeezing/Fock/SBS/noise-transfer/cross-fidelity suite：
  `257 passed`（中间回归；后续全量再次覆盖）；
- protocol/parameter registry：`41 passed`；
- quadrature machine contract：15/15 PASS；
- cross-fidelity：15/15 PASS；
- 10/12 dB 最大 canonical Fock `|q-p|` LER gap：`1.51e-7`；
- legacy ambiguous 高 squeezing 最小 p-q gap：`0.4182`；
- 12 dB Fock/direct q-LER gap：`4.61e-4`，仍由 cutoff tail 主导；
- 3 dB noise/direct gap：`0.01541`，仍作 clipping falsification；
- full `tests/`：`751 passed, 4 failed`；四项均为既有 R-N012 缺失 FR8/P4 文档，
  与本 task 无关。

## 非 demo 复核

复核发现并修复了“只改 folding period”的潜在伪实现：若不同时缩 width、envelope、
Jacobian 与 variance，q/p 可以局部过门但不代表同一物理态。新增测试直接比较四逻辑态
canonical wavefunction 与 decoder dilation、canonical source 与 registered Fock coefficients，
并对 mixed-coordinate contract 主动报错。

## 风险与 claim 影响

- R-N041：`Open/Critical/Immediate -> Mitigated/Medium/Monitor`；坐标与 axis-resolved
  Fourier 问题关闭，legacy negative provenance 保留。
- PC-N01：coordinate/analytic 子门通过；source-device envelope、`nbar`、state-moment
  calibration 仍为 `null`。
- R-N037/R-N038/R-N040：数值回灌但不关闭，finite cutoff、completed Kraus 与
  middle-fidelity surrogate 边界不变。
- 允许 axis-resolved canonical q/p 与独立轴 Pauli projection；禁止升级 coherent
  correlated joint-axis/process fidelity、device calibration 或无限维结论。

## 是否插入新 task

不插入。剩余项已有 T2.3.4、T5.3、T6 和 PC-N01 device-calibration gate 承接，不阻塞
下一正常任务。

## 任务板同步

- T-RISK-20260714-01：`In Progress -> Done`；
- T2.3.4：`Todo -> In Progress`；
- 当前推荐任务：`T2.3.4`。
