# T2.1.2 observed-only 多轮控制 memory

**日期：** 2026-07-14  
**实现：** `physics/control_memory.py`  
**协议：** `PROTO-SBS-MAIN`  
**证据范围：** deterministic observed-only memory contract；不是完整 controller/fallback policy 或硬件 runtime

## 1. 现有代码审计与实现位置

旧 `FastLoopEmulator` 已各自保存 `_cumulative_residual`、当前 correction 和 active `ParamBank` version，`SBSObservationResetModel` 已保存 e/leakage runs，`SBSErrorSpaceInstrument` 已定义 `PauliFrame`；但这些状态分散在不同层，且没有统一保存 confidence、phase frame、deadline flag，也不能直接接 T2.1.1 的 observed stream。

T2.1.2 没有重写旧 fast loop，而是新增可复用的协议层 `MultiRoundControlMemory`：输入类型严格为 `ObservedSyndromeStep + ControlDecision`，复用 shared `LATTICE_CONST`、`PauliFrame`、`PairedSyndrome` 和现有 `ParamBank.active_version` 语义。后续 runtime 可把 fast-path 的实际 action/version/deadline 适配成 `ControlDecision`。

## 2. Memory state

`ControlMemoryState` 每周期保存：

- `accumulated_residual_shift=(q,p)`：当前 modular observation 的连续 nearest lift 减去本周期实际 correction 后的 post-action estimate；
- `previous_correction=(q,p)`：本周期实际执行 correction command；
- `confidence=(q,p)` 与 `minimum_confidence`；
- GF(2) `PauliFrame(x,z)`；
- 规范化到 `[-pi,pi)` 的 X/Z `phase_frame_rad`；
- 分离的 `x_e_run/z_e_run/leakage_run`；
- 单调 `parameter_bank_version`；
- 当前 `deadline_missed`、连续 `deadline_miss_run` 和累计 `deadline_miss_count`；
- `cycle_index/cycle_count`。

所有 state/decision 都是 frozen dataclass，pair/probability/bit/version/boolean/text 均 fail closed；counter 有显式 `counter_max` 饱和。

## 3. Residual continuity 与 correction 符号

T2.1.1 的 residual 在 `[-lambda/2,lambda/2)` 内。memory 以上一周期 post-action estimate `a_{t-1}` 为参考，对当前 modular residual `s_t` 选择最近 lift：

\[
k_t=\left\lfloor\frac{a_{t-1}-s_t}{\lambda}+\frac12\right\rfloor,
\qquad
\tilde s_t=s_t+k_t\lambda.
\]

随后沿用仓库 `LogicalErrorTracker/FastLoopEmulator` 的 correction 符号：

\[
a_t=\tilde s_t-c_t,
\]

即正 residual 由正 correction command 抵消。初版审查曾发现“把 correction 当作直接相加物理位移”的符号自洽但跨模块错误，现已由现有 tracker 源码和 direct test 锁定。

nearest-lift 保留跨 wrap boundary 的连续性，例如 `0.49 lambda -> -0.49 lambda` 被解释为 `0.49 lambda -> 0.51 lambda`，alias index 为 1。该方法隐含相邻有效观测之间的可辨识连续性假设；真实变化超过半格时可能发生 cycle slip，已登记为 R-N032。

## 4. Frame、run、version 与 deadline 语义

- Pauli frame 只由 deployable decision 的 `pauli_frame_delta` 做 GF(2) XOR；禁止从 true logical label 更新；
- phase frame 对 decision delta 相加后 wrap 到半开区间；
- e/leakage runs 由 observation class重新计算，不盲信输入 counter；默认把 T2.1.1 提供的 counter 当冗余一致性校验，不一致即拒绝；
- parameter-bank version 必须非递减，version change 在 update record 中显式标记；真实 `ParamBank` stage/commit 的 `0->1` 集成已测试；
- deadline miss 是本周期实际状态，不自动清零 correction。原因是 deadline miss 后仍可能执行 local safe fallback；`ControlDecision.applied_correction` 必须记录最终真实动作。T4.2/T4.3 才定义 miss→hold/fallback/recovery 的执行策略。

## 5. Schema 与事务边界

`MultiRoundControlMemory.update` 只接受 `ObservedSyndromeStep`，传入包含 truth 的 `SyndromeStreamStep` 会类型拒绝。deployable record 不含 `DriftState`、hidden regime、outlier component、leakage truth、logical label 或 oracle 字段。

每次 update 在全部 cycle/order、validity、analog-residual、run counter 和 bank rollback 检查通过后才替换 state；失败不留下半更新。`run()` 是流式 append，不把一批输入伪装成数据库事务；调用者可用 `reset()` 回到显式 initial state。

## 6. 反 demo 验证

`tests/test_control_memory.py` 的 26 项测试覆盖：

1. config、decision、state、type、finite/probability/boolean/version 失败分支；
2. 全部 required memory 字段与 public export；
3. wrap boundary nearest-lift 与 alias index；
4. 与现有 fast-loop 一致的 correction subtraction；
5. confidence、Pauli XOR、phase modulo；
6. X/Z e-run 分离、任一 constituent leakage run、counter saturation；
7. counter mismatch、invalid observation、analog/residual mismatch、cycle replay/skip 的 transactional rejection；
8. parameter-bank rollback rejection与真实 `ParamBank` atomic commit version；
9. deadline current/run/total 与 nonzero local-safe fallback action；
10. T2.1.1 observed stream 集成和 full truth-step rejection；
11. deployable schema hidden-key audit；
12. reset、分段 append 与 fresh-instance exact replay。

当前未实现 confidence calibration、nearest-lift cycle-slip detector、CRC/age/CAS/ack、deadline enforcement 或 fallback FSM。这些分别由 T3/T4/T5 的既有任务承接，不能从 memory contract 外推为已完成控制器。
