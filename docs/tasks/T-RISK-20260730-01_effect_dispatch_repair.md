# T-RISK-20260730-01：T06 effect dispatch 修复（legacy mirror）

- **日期**：2026-07-30
- **状态**：Done
- **权威完成记录**：
  `docs/new_tasks/T-RISK-20260730-01_effect_dispatch_repair.md`

## 输入与实际完成

本任务消费 T06 V1 的8个 selection chunks、24个 confirmation chunks、
7,168个 density trials、8,404-row Source Data 与3,043-gate blueprint。
修复只由 canonical cell identity 恢复注册 effect 类别，derived
`true_distance` 继续作为独立观测值核对；没有重抽、改seed/count/factor/margin、
删门或改变 maxT 相关结构。

三条实现失败 lineage 均永久归档：

- V1 exact-float dispatch failure；
- multithread maxT serialization failure；
- verifier source-derived effect input failure。

最终 V1→V2 只有六个注册 `effect_0.050` summary 的
`gate_pass/power_gate_pass` 两字段变化；32个 raw chunks、其余 Source Data、
maxT字符串和 selected blueprint不变。

## 产物与验证

- repair verdict=`PASS_ANALYSIS_ONLY_EFFECT_DISPATCH_REPAIR`；
- repair analysis=
  `d8f965fbf0eb8f38e519f40dfd84808cf03e52100bdece7fd61e7d3310c58ca0`；
- 独立 verifier=`21/21`，analysis=
  `5a49967fddc86283a42ff75091ce099eed9d3ae799763a3d1e4cca40c7a19c2e`；
- `78 passed in 19.24s`，144项 live evidence audit 与 py_compile通过；
- repair report自身 final release=false，最终 release只由独立 verifier签发。

## 风险、插入任务与任务板同步

- R-N195--R-N197 降为 Mitigated / Monitor；
- 不插入新任务，恢复 T-RISK-20260728-04；
- T04只获得 preregistration，所有 scientific/performance/physical/hardware/
  official/Puviani/SOTA claim仍为null；
- `docs/new_task_board.md` 中本任务状态为 Done。
