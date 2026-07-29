# T-RISK-20260728-06：完整 gate-union 的有限 count 选择与独立确认

- **日期**：2026-07-30
- **状态**：Done（independently verified count selection + confirmation PASS）
- **来源风险**：R-N185、R-N192、R-N193、R-N194、R-N195
- **前置任务**：T-RISK-20260728-05；implementation repair T-RISK-20260730-01
- **后续任务**：T-RISK-20260728-04

## 输入材料

- T05 independently verified statistical NO-GO；
- 完整 3,043-gate blueprint：3,037 stochastic + 6 deterministic；
- T05 证明旧 `round=768 / fault=384 per state` 的 joint-null global power
  仅 `0.433`、LCB=`0.4177`；
- 预冻结候选 scale `{1.5, 2.0, 2.5, 3.0}`；
- 互斥且单射的 selection/confirmation density 与 maxT seed namespaces；
- paired-density factor=`1.0`、`B=199`、higher quantile、完整 closed-family
  joint maxT 与 IUT/TOST 功效规则。

## 执行方案

1. 不访问任何 T04 formal outcome，先冻结 finite candidate grid、最小通过规则、
   候选耗尽 NO-GO 与资源上限。
2. selection split 对四个候选同时运行 d120/d132 heteroskedastic local density
   和完整3,043门 joint-maxT power。
3. 只选择同时通过两个门的最小候选；不得直接翻倍、事后删门、拆 family 或改 margin。
4. untouched confirmation 使用与 selection 完全互斥的 seed，对选中 count 执行
   3 families × 2 dimensions × 4 effects 的24-cell density确认，并重做完整
   joint-maxT/power。
5. writer 只负责生成初步 count selection/confirmation report；physics-free
   独立 verifier 从 seed 重建全部 7,168 density trials、1,195 maxT/power
   cases 和4个 linked blueprints。
6. 任何确认门、资源门、binding、seed、raw、summary 或 claim 边界失败均 fail-closed。

## 实际完成内容

### 1. Selection

- 8 个 density chunks、1,024 raw trials；
- 四个候选均重建完整 linked blueprint 与 joint maxT；
- selection maxT replicates=`4×199=796`；
- selection maxT power cases=`4×40=160`；
- scale=`1.5` 未通过完整 joint门；
- scale=`2.0`、`2.5`、`3.0` 通过；
- 按预注册最小通过规则选择 scale=`2.0`。

选中计数：

- state clusters=`768`；
- round clusters=`1,536`；
- aggregate fault trajectories=`4,608`
  （`768/state × 6 states`）。

### 2. Confirmation

- 24 个 confirmation chunks；
- 24 cells × 256 trials=`6,144` density raw rows；
- 三个 family：low-energy balanced、heteroskedastic coherent、
  heavy-tail rare coherent；
- dimensions=`{120,132}`；
- effects=`{0,0.05,0.1,0.12}`；
- confirmation maxT replicates=`199`；
- confirmation maxT power cases=`40`；
- 3,043-gate selected blueprint 保持3,037 stochastic + 6 deterministic；
- writer 在 T-RISK-20260730-01 最小 correction 后为 `15/15`，
  verdict=`PASS_COUNT_SELECTED_AND_UNTOUCHED_CONFIRMED`。

由于 confirmation summary 的 categorical dispatch 在首次 writer 中存在实现缺陷，
最终结果必须准确表述为：

> frozen-sample post-outcome implementation correction

不得不加说明地声称从实现层面全程 untouched；raw sample、seed 和统计合同本身仍保持 untouched。

### 3. Source Data 与资源合同

Source Data 共8,404行：

- density raw：7,168（selection 1,024 + confirmation 6,144）；
- maxT replicate：995（selection 796 + confirmation 199）；
- maxT power：200（selection 160 + confirmation 40）；
- density summary：32（selection 8 + confirmation 24）；
- selection summary：4；
- meta：5。

T04 预注册资源预测：

- chunks=`518`；
- exact rows=`2,085,888`；
- wall estimate=`460,880.19760518876 s`；
- artifact estimate=`117,026,773,870 bytes`；
- peak RSS estimate=`17,128,488,960 bytes`；
- T04 仍必须在任何 scientific chunk 前运行 fresh 四层/c44-B/RB-compression
  resource benchmark，不能直接迁移本预测。

## 独立验证

完整独立 verifier 于 2026-07-30 01:29:58 启动，并于03:38:55正常结束：

- 使用单独进程与四个单线程 worker；
- 不导入 writer、physics、prior verifier 或 paired-UQ helper；
- 完成 selection 与 confirmation 全量重算；
- gate ledger=`21/21`；
- verdict=`PASS_INDEPENDENT_COUNT_SELECTION_AND_CONFIRMATION`；
- selection density rows=`1,024`、confirmation density rows=`6,144`；
- linked blueprints=`4`；
- selection maxT=`796` replicates / `160` power cases；
- confirmation maxT=`199` replicates / `40` power cases；
- maximum numeric delta=`1.942890293094024e-16`；
- analysis SHA256=
  `5a49967fddc86283a42ff75091ce099eed9d3ae799763a3d1e4cca40c7a19c2e`；
- verification file SHA256=
  `ffcb66d4ca204e8fdbe7c9e10a0e46d6676a0e7fc8eb9a72c0af53bf14b4513c`；
- stdout JSON 与 atomic report 语义相同，stderr=`0 bytes`；
- 没有重复 supervisor/verifier，主进程与四个 child 正常退出；
- `t04_preregistration_released=true`，但
  `t04_scientific_execution_released=false`；
- qualified claim 与 twin/LER/lifetime/physical/hardware/official/Puviani/SOTA
  八类字段全部保持 `null`。

## 产物路径

- `configs/phase9/t_risk_20260728_06_count_selection_confirmation.json`
- `cnn_fpga/benchmark/phase9_count_selection_confirmation.py`
- `cnn_fpga/benchmark/phase9_count_selection_confirmation_verify.py`
- `cnn_fpga/benchmark/phase9_count_effect_dispatch_repair.py`
- `tests/test_phase9_count_selection_confirmation.py`
- `tests/test_phase9_count_selection_confirmation_verify.py`
- `tests/test_phase9_count_effect_dispatch_repair.py`
- `runs/t_risk_20260728_06_count_selection_confirmation/`
- `docs/t_risk_20260728_06_count_selection_confirmation.json`
- `docs/t_risk_20260728_06_count_selection_confirmation_source_data.csv`
- `docs/t_risk_20260728_06_selected_gate_blueprint.json`
- `docs/t_risk_20260728_06_count_selection_confirmation_verification.json`

当前关键证据：

- writer report analysis=
  `576e33b31e9e540bf0be4b6d0fd032f930bfbd276a73b05af2285f1a2fed2a01`；
- Source Data：1,985,525 bytes，SHA256=
  `0ebeb19ed98dc91059c46814bdc68c0b666692d2b9d34b93de8d52831a29f8be`；
- selected blueprint：1,290,746 bytes，SHA256=
  `9bd9ed23fb30737390bacc69f2730755add80257a3814475e12b7f13c7d1409b`；
- selected blueprint analysis=
  `53a24463446874b055ac462d26b3608a5ebc7729f1555e200e9a9ebc13f4d193`。

终态验证：

- focused writer/repair/verifier tests：`78 passed in 19.24s`；
- py_compile：writer、repair、verifier 与三份测试源码全部通过；
- 独立 live evidence audit：144项检查全部通过；
- 三个 failure archive manifest 自哈希有效；
- V1 local archive 38/38 files、7,548,008 bytes 与 canonical digest有效；
- live 32/32 raw chunks 与 V1逐字节一致；
- 8,404-row CSV inventory与六行×两字段白名单精确；
- selected blueprint与V1逐字节一致且3,043 gate ID唯一。

## 反简化审计

- 没有把 T05 的 synthetic power influences 当作 T04 scientific outcome；
- 没有把3,043门缩成旧1,589门或按 family 拆开后拼接通过；
- 没有用 pointwise z 替代相关 joint maxT；
- 没有只运行最可能通过的 scale=2.0：selection 完整覆盖四个有限候选；
- 没有用 selection seed 做 confirmation；
- 没有只看平均 density：confirmation 覆盖3 family×2 dimension×4 effect；
- 没有把 exact-float 实现失败登记为科学 NO-GO；
- 没有重抽、追加样本或放宽阈值修复 implementation failure；
- 完整独立 verifier 从 seed 重算，而不是只比较 writer summary；
- T06 只选择 T04 的预注册 count，不证明 twin、LER、lifetime、
  physical break-even、hardware measured、official Puviani 或 external SOTA。

## 风险复核与插入任务

- R-N185：selected count 的高维 density local power 已由 disjoint confirmation
  与独立重算闭合，降为 Mitigated；不迁移为 T04 twin PASS；
- R-N192：完整3,043门 joint-maxT 的设计功效已由独立 confirmation闭合，
  降为 Mitigated；真实 T04 outcome 仍须使用 observed influences；
- R-N193：继续 Open；T04 的真实 raw/archive/finalize 资源峰值尚未测量；
- R-N194：finite selection + disjoint confirmation 已独立 PASS，降为 Mitigated；
- R-N195--R-N197：由 T-RISK-20260730-01 完成并降为 Mitigated；
- 当前不插入新科学 task；verifier失败时必须按 failure class 决定是科学 NO-GO、
  infrastructure incomplete 还是 analysis-only implementation repair。

## 对任务板的同步

- `T-RISK-20260730-01 = Done`；
- `T-RISK-20260728-06 = Done（Independent PASS）`；
- `T-RISK-20260728-04 = In Progress`；
- 即使 T06 PASS，T04 scientific execution 与全部性能/物理/硬件/official/Puviani/SOTA
  claim 仍保持 blocked/null。
