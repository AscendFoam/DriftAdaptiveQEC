# Route-A regime-aware 安全自适应策略

## 1. 当前结论与证据边界

T6.6.2 已升级为与 T6.6.3 V4 一致的 **Window/EWMA 双影子 bank** 软件执行策略。它把
observed posterior、prequential bank routing、原子 A/B bank、5+1-cycle event/action、
leakage/reset 与 integrity-only LKG rollback 接到同一条 production-cadence 路径。

本 task 的 20,061-cycle posterior 仍是结构覆盖 fixture，不是 T6.6.3 校准 HMM 的统计输出；因此这里只
证明结构、因果性、预算、版本和故障语义，不证明 LER/tail 优势、integrated RTL timing 或板测性能。

## 2. V4 状态与 bank 行为

| 输入条件 | 可提交的 bank | 失败行为与恢复 |
| --- | --- | --- |
| normal/smooth 且 policy open | Window 只有在 `smooth posterior >= 0.30` 且上一完整窗口的 pre-update predictive NLL 严格优于 EWMA 时才可提交；否则提交 EWMA | 完整 CRC/SHA、CAS、ack/readback 与 4,000-cycle residency 均不可绕过 |
| calibration-shift/burst | normal adaptive promotion 冻结；只允许 continuously updated、validated EWMA shadow | policy 明确记为 fallback/tail；不再回切初始 static，也不冻结 EWMA 到 stale |
| posterior uncertain | 只允许 validated EWMA shadow | 不把统计不确定性伪装成 integrity fault，也不无条件回滚旧 LKG |
| leakage/reset | 禁止参数提交 | event FSM 发出 reset；ack 后仍需健康/posterior hysteresis |
| CRC/SHA/version/age/deadline/router-hash 异常 | 禁止 Window/EWMA candidate | 立即 fail closed，并以更高版本 republish monotonic LKG |

每个 candidate 必须携带 Window/EWMA prequential scores、router algorithm SHA，以及 **完整双影子系统**
成本，而不是只申报被选中的单专家成本。当前冻结成本为 `1,218/8,192 MAC`、private state
`3,840/8,192 bytes`、workspace `3,072/8,192 bytes`。

## 3. 反简化实现

1. Window/EWMA 都在相同的完整 observed parameter window 上生成 shadow；Window 只能在下一 parameter
   period 使用，不读取当前/未来 observation。
2. tail/uncertain 可以提交 EWMA shadow，但 pending Window 会被真实 hysteresis-invalidated；integrity/reset
   会取消所有算法 candidate。
3. `online_update_frozen` 表示普通 adaptive promotion 冻结；EWMA trusted-shadow promotion 有独立
   `candidate_accepted`、decision source 与 trusted-switch provenance，避免“冻结”字段掩盖安全更新。
4. LKG rollback 保留 semantic content、生成更高版本 integrity identity；被拒绝的未激活版本只有从
   registry 清除后才能安全复用。
5. router hash、smooth posterior、score ordering、总系统 MAC、fixed-point image、source cadence 任一不符都
   fail closed。负向单测还覆盖低 smooth posterior、Window score 不胜与少报总成本。

## 4. production-cadence 长轨

`python -m cnn_fpga.benchmark.regime_aware_safe_policy_validation` 运行 cycles 5--20,065：

- 20/20 machine gates、11/11 mutations，20,061 行 Source Data；
- 每个 action 的 source-to-action 都是 6 cycles；
- Window/EWMA/Window/待取消 Window candidates 分别在 4,000/8,000/12,000/16,000 cycles 到达；
- confirmed commits 为 versions `1,2,3,4`，cycles `4001,8002,12003,16005`；
- tail 中 8,000-cycle EWMA shadow 被允许提交，8,002 起 decision source 为
  `trusted_ewma_shadow`，没有回切初始 static；
- 16,000-cycle pending Window 在 16,001-cycle integrity fault 下被拒绝；16,005 以相同 next version
  完成 guarded LKG rollback；
- residency 由相邻正常 commit 间隔直接验证为至少 4,000 cycles；实际 deferred cycles 为 6，旧版
  tail→static 导致的 3,942-cycle 无效等待已被架构消除，而不是通过缩短合同掩盖。

## 5. 产物

- `cnn_fpga/runtime/regime_aware_safe_policy.py`
- `cnn_fpga/benchmark/regime_aware_safe_policy_validation.py`
- `docs/t6_6_2_regime_aware_safe_policy.json`
- `docs/t6_6_2_regime_aware_safe_policy_source_data.csv`
- `tests/test_regime_aware_safe_policy.py`
- `tests/test_regime_aware_safe_policy_validation.py`
