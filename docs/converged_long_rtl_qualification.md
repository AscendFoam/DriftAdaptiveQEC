# T6.25.3 converged top 百万周期 CXXRTL 资格验证

## 结论

**`PASS_EXACT_CONVERGED_TOP_MILLION_CYCLE_CXXRTL_QUALIFICATION`**。10 个 family 各 100,000 cycles，聚合 1,000,000 cycles；新 trace 对 T6.25.2 exact converged top 的全部 148-byte 公开输出向量逐周期比较，bit mismatch、undefined action、CRC error、silent overflow 与 silent version wrap 均为 0。

## 硬件执行合同

- source-to-action 恰为 6 cycles，MAP debug 恰为 5 cycles；连续输入的 II=1 pair 为 998,435，输出 pair 数相同，无 bubble。
- 完整镜像事务实际传输 257×2 个 22-bit words，覆盖 CRC32、inactive write、trust、host/policy commit、cancel、drain、snapshot 与全部 11 类 reject reason。
- 可从封装端注入的 core fault 位均被命中并恢复；由 converged manager/数据通路结构排除的 fault 位始终为 0，未通过重建 raw bypass 伪造覆盖。
- CXXRTL comparator 对 148 个 expected bytes 逐字节 shadow mutation，148/148 被检测；21/21 report semantic mutations 被独立 gate 重算拒绝。
- 版本长轨无下降；near-wrap 不是靠百万周期从 0 暴力递增，而是绑定同一源码 T6.25.2 的 actual-core arbitrary-state atomic proof 与 near-wrap witness。

## 边界

这仍是 two-state、pre-board CXXRTL。真实 transport/CDC/pins/bitstream、板测 latency/jitter/deadline/power、跨工作 fastest/SOTA 与 multimode decoder in RTL 均未建立。
