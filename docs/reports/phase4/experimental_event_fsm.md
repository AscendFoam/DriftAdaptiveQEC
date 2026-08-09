# T4.2.2 实验式事件状态机与硬件动作

## 1. 范围与接口

本任务把 T4.2.1 的 version-bound MAP decision 接到 observed-only 事件状态机。在线输入只有当前
`g/e/leakage` X/Z 观测、quadrature phase、对齐的 MAP decision、active bank version、reset ack 以及
valid/CRC/fresh/deadline health flags；不接收 `DriftState`、logical truth、hidden recovery depth 或未来观测。

旧 `RunLengthParameterBankFSM` 保留为 T3.2.5 负基线；新实现是
`ExperimentalEventFSM`，避免把参数更新 baseline 静默升级成实验 fast-path contract。

## 2. 状态、优先级与失败分支

FSM 有 `normal/x_recovery/z_recovery/hold/reset_request/fallback` 六态，维护六个 3-bit 饱和计数器：
X/Z `e` run、leakage run、post-leakage clean run、health-good run 和 reset-wait run。默认优先级为：

1. health fault 立即进入 `fallback`；非请求态收到 reset ack 也 fail closed；
2. `reset_request` 在 ack 前保持 sticky；ack 后进入 `hold`；
3. 单次 leakage 进入 `hold`，连续两次 leakage 请求 reset；
4. hold/fallback 分别需要两个 clean/good cycle 才退出；
5. X/Z `e` 连续两次进入对应 recovery，同时满足时由当前 phase 决定确定性优先方向；
6. 其余进入 `normal`。

所有输入一致性检查在状态修改前完成。cycle gap/replay、MAP-valid 对齐、phase/version 不匹配、version
回滚、logical-action/LLR-sign 不一致都会抛错，且 state/history 保持不变。

## 3. Frame 与时序合同

非安全态的 MAP flip 原子更新对应轴的 GF(2) Pauli frame，并给 8-bit phase-frame 加半圈码 `128`
（模 256）。这里的 phase-frame 只是逻辑表示镜像，不是实际微波相位、脉冲或物理 recovery。`hold`、
`reset_request`、`fallback` 禁止 correction，抑制 pending flip，四个 frame delta 全为零。

T4.2.1 MAP decision 在 source 后第 5 cycle 有效，FSM 再经 1-cycle action register，因此硬件动作合同为
source 后第 6 cycle，initiation interval 为 1。该数字来自 software cycle model，不是 RTL、综合、post-route
或实板时序。

## 4. Production 验证

8 个确定性场景各 128 cycles，共 1,024-row Source Data：nominal frame、X/Z recovery saturation、双轴
`e` phase tie、leakage/reset handshake、四类 health fault、六计数器 saturation、8-bank version switch。
20/20 gates 覆盖全部六态、要求的 mode transitions、counter saturation、两种 tie-break、sticky reset/ack、
fallback hysteresis、安全态 frame 不变、双轴 frame/modulo wrap、精确 6-cycle latency、II=1、image hash/version
一致、transactional negatives、确定性 replay 和 truth-field denylist。

bank 0 是真实注册的弱 static profile，其量化支持全为 `I`。基准不伪造不可达 flip；bank-switch 场景保留
该负事实，frame action coverage 由确实同时含 `I/flip` 的 bank 1 完成。

## 5. 证据边界

当前只允许称“observed-only six-mode integer event/frame software contract connected to version-bound MAP
decisions”。最小 live-state 代理为 55 bits，但不是综合资源估计；LUT/FF/BRAM/DSP/Fmax、RTL measured 和
board measured 字段保持 `null/false`。本任务没有证明 device-calibrated recovery/reset 有效性，也没有完成
T4.2.3 的完整 conservative fallback、T4.2.4 的端到端定点、LER impact 或任何 FPGA/board claim。

