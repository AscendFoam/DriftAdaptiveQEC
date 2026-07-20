# Phase 6D multimode 确认性预注册

> seal：`52aa0a9106b75581166c117fa8c4439c5f802a0a888ba29ecebf58d591b9c37e`；verdict：`PASS_PHASE6D_MULTIMODE_PREREGISTRATION_SEALED`。T6.18.3 仅作 opened development，不进入本 formal。

## 四分割

| Split | 独立 clusters | d | sigma | 每 family rounds | 角色 |
| --- | ---: | --- | --- | ---: | --- |
| `train` | 24 | [3, 5] | [0.48, 0.54] | 2048 | fit posterior/approximation parameters only; no claim, threshold or candidate selection |
| `calibration` | 16 | [3, 5] | [0.5, 0.57] | 3072 | freeze posterior calibration, likelihood truncation and finite candidate grid without choosing a winner |
| `pilot` | 24 | [3, 5] | [0.52, 0.6] | 4096 | select exactly one candidate and strongest eligible deployable baseline once |
| `formal` | 60 | [3, 5] | [0.46, 0.56, 0.62] | 4096 | untouched confirmatory evaluation; no fit, calibration, threshold change, candidate replacement or denominator change |

## 功效与 formal 规模

- 60 个 formal seed-cluster；13 个 family；每方法 3,194,880 physical rounds。
- 12-comparator Bonferroni 一侧设计：required clusters=48，planned=60，approximate power=0.9631。
- 功效计算只用于冻结 N；正式结论只由 paired simultaneous bootstrap CI 决定，pilot 后不扩样。

## 统计、tail 与缺失处理

- `seed; resample all methods, distances, sigma strata, families and windows for that seed together`；bootstrap=50,000。
- SOTA 门：relative LER simultaneous 95% LCB `>10%`，absolute LCB `>0`。
- calibration/telegraph worst-window 与 CVaR95 独立成 family；stationary/OOD margin 不得由 aggregate 掩盖。
- proposed failure 保守计错并使 integrity gate 失败；required baseline failure 关闭 SOTA，不得删除对手；零填补禁止。

## 不可变与访问规则

- train→calibration→pilot→formal 单向；pilot 只选一次，锁 candidate/baseline/checkpoint/config hash 后才可打开 formal。
- outcome/significance/precision 不得触发 early stop 或扩样；formal 只允许资源、完整性、数值或工具失败终止。
- seal 后 amendment 只能修正文案或 locator，`affects_analysis=false`；分析字段变化必须新建前瞻协议，v1 结果不能救援。

## Opened-data 隔离

- seed overlap=[]；sigma overlap=[]；spatial fixed-pattern matches=[]。
- 全部 cross-split factor overlap 均为空：True。
