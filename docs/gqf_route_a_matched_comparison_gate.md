# T6.8.5：Route-A 与 Puviani NMF matched comparison 前置门

## 结论

T6.8.5 按预注册失败分支完成：`COMPLETE_T6_8_5_INELIGIBLE_NEGATIVE_BRANCH`。

T6.8.4 的 official paper-exact qualification 为 0/15；official NMF/MF checkpoints、20-agent seeds、
selection ledger、six-state 1000-cycle evaluator、raw trajectories 和 matched training/search budget 均不存在。
因此同一 GQF simulator/budget/selection/seeds 下的 Route-A vs NMF comparison 目前不可定义。

## 执行边界

- 8/8 mandatory prerequisites 均显式评估为 false；
- comparison run manifest 与 raw data 均为 `null`；
- lifetime、paired improvement/95% LCB、gain retention、params/MAC/memory、fallback 与 unsafe-action
  共 13 个字段全部为 `null`；
- 不使用 T2.3.7/T4.4 的不同 simulator teacher/student 结果代替 official NMF；
- 只有作者发布 paper-matching implementation、20 checkpoints/seeds、selection ledger、six-state raw
  evaluation、完整 fit contract，并具备可运行 exact compute 后才可恢复。

## 验证

- 10/10 integrity gates；
- 10/10 target-specific semantic mutations；
- GQF intake/exact/matched-gate/board tests 合计 `26 passed`；
- parent JSON、gate implementation 与 8-row Source Data 均 SHA-256 绑定。

当前只允许写：因 official exact 前置失败而没有运行 matched comparison。禁止 same-GQF lifetime、paired
improvement、surpass Puviani NMF，以及把独立 student 结果称作 official-GQF extension。

