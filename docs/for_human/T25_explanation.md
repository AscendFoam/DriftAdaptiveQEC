# T25 说明

这次 `T25` 做的是只读 gate review，不是重新跑 benchmark。

结论很简单：

1. `T24` 的结果可以当作一次**已完成的 frozen-set formal software revalidation**
2. 这个结论只成立在 **mock-backed software HIL** 边界内
3. 它**不能**被写成 `.tflite` runtime 已恢复、`real_board` 已验证，或者 paper-grade 扩展 benchmark 已完成

为什么还能通过：

- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743` 的 evidence pack 是完整的
- `missing_runs = []`
- 20/20 scenario/mode rows `coverage = 1.0`
- 40 个 repeat-runs 都完成了
- 四个场景下 winner 都是 `hybrid_residual_b`

为什么仍然带 warning：

1. `correction_saturation_rate_mean` 在全部 20 行里都是 `0.0`
   - 现在还不能判断这是真实结果，还是指标采集路径没有打通
2. `teacher_scalar_diagnostics.csv` 只有表头，没有数据行
   - 这说明 teacher diagnostics 这条机制证据链还没收口

所以，`T24` 现在能支持的是：

- “历史 frozen-set P4 软件 benchmark 在当前恢复路径上重新跑通了”

不能支持的是：

- “部署链已经恢复”
- “真板路径已经验证”
- “teacher 机制已经解释清楚了”

当前建议的下一优先任务类型是机制证据审计，优先看 `teacher diagnostics` 路径，再判断是否与 `correction saturation` 一起收口。
