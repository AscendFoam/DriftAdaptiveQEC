# T4.2.3 Conservative fallback 与健康标志

## 1. 设计目标

T4.2.2 已能在局部 valid/CRC/fresh/deadline fault 下进入 `fallback`，但仍假定每周期存在结构正确且已
对齐的 MAP decision，也没有 OOD、image readback、unknown/mismatch/rollback version 和逐原因计数。
T4.2.3 在 event FSM 前新增 `ConservativeFallbackController`，把 operational fault 转成可继续逐周期推进的
frame-hold/reset action，而不是让缺失或损坏 decision 使 fast path 直接异常退出。

## 2. 可信输入与 14-bit health registry

controller 初始化时接收只读 `TrustedParameterImage(version, CRC32, SHA256)` registry；周期输入只能报告
当前 readback，不能自行定义可信值。14 个固定优先级标志为：observation invalid、OOD score exceeded、
input CRC、image CRC、image SHA、unknown version、decision/version mismatch、version rollback、parameter stale、
deadline miss、MAP missing、MAP alignment/action invalid、unexpected reset ack 和 leakage observed。

默认 OOD 是 8-bit code，`<=192` 通过、`>192` 回退；parameter age `<=64` 通过、`>64` stale。这两个阈值
是当前 software contract，不是 device-calibrated OOD 或物理 freshness 结论。每个标志都有固定 bit、
uint8 saturating per-flag cycle counter，输出同时保留完整 flag tuple、bitmask、primary reason 和 FSM reason。

## 3. Conservative action

- 任一 blocking fault：拒绝本周期 MAP，不接受新 version，输出 `frame_hold_no_map`；Pauli/phase-frame delta
  全为零。若 reported version rollback/unknown/mismatch，last trusted version 保持不变。
- leakage 是非 blocking event class：validated MAP 可以完成 integrity accounting，但 T4.2.2 的 hold/reset
  mode 会抑制其 action；连续两次 leakage 请求 reset，ack 后 hold。
- fault 后需要两个 good cycles 才退出：第一周期仍为 `recovering` 且 frame hold，第二周期才恢复 map action。
- 结构级 API 错误（cycle gap、字段类型、超出 OOD word width）在任何 state/history 修改前拒绝。

这里的 frame hold 是“本周期不使用未经验证的 correction”，不是自动切回 bank 0，也不是物理 reset/
recovery 有效性的证明。自动 bank rollback、transport watchdog/readback transaction 属于 T4.3/T6 后续范围。

## 4. Production 证据

16 个场景各 256 cycles，共 4,096-row Source Data。20/20 gates 验证全部 14 flags、blocking fault 的
frame/map 不变性、leakage reset/ack、OOD/age 边界、CRC/SHA 区分、8 个 version 单调接纳、rollback/
unknown/mismatch 保持 trusted bank、组合 bitmask/reason trace、两周期恢复、uint8 saturation、双轴健康
MAP action、精确 6-cycle latency、II=1、deterministic replay 和 no-truth schema。

## 5. 证据边界

允许表述为“traceable observed-health and integrity frame-hold/reset software fallback contract”。readback
CRC/SHA 被假定由上游对实际 image 重新计算；当前代码不证明片上连续 scrubbing。OOD score 的产生、校准、
false-positive/false-negative、物理 LER impact 尚未验证。资源只报告 exact state/reason-word proxy，LUT/FF/
BRAM/DSP/Fmax、RTL measured、board measured 继续为 `null/false`。

