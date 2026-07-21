# T6.25.2 converged production RTL property/cover/mutation

## 结论

**`PASS_CONVERGED_PRODUCTION_TOP_PROPERTY_COVER_MUTATION_CLOSED`**。唯一生产顶层同时包含参数管理器、Route-A 提交仲裁、六周期 single-mode 核心与 policy/LKG overlay；外部不暴露 raw `cfg_we` 或 `bank*_trusted`。17/17 gates、14/14 reachable witnesses、21/21 targeted RTL mutations 通过。

## 证明边界

- unbounded safety：k-induction 先闭合 reset-reachable 管理/策略不变量与 present-state guards；随后在任意满足这些已证不变量的 predecessor state 上证明全部 transition assertions。
- bounded regression：从同步复位出发 20 cycles 的独立 BMC 继续覆盖 CRC32、ordered full image、trust/version/CAS、old-or-new、cancel/drain/conflict/backpressure、LKG 与 near-wrap；formal image 深度缩为每相位 2 words，仅用于让完整事务可达。
- actual core：真实 `gkp_fast_path_core` 的 deadline 与 age 两个 II=1 样本连续六周期输出，均显式 fallback、零 action/frame delta；另以任意 predecessor state 证明 ACK、bank 与 activation version 更新严格细化 manager 使用的原子提交契约。
- monolithic combined k-induction 的旧失败尝试没有升级为证据；当前只声称已分解闭合的 unbounded safety，不声称 unbounded liveness/fairness。

形式化首先发现并修复了一个非演示级缺陷：active bank/version 已切换而注册 ACK 尚未返回时，旧 manager 会重复呈现 commit 一周期。现在 core-facing 输出重新检查 boundary、target、plus-one、no-wrap 与 trust。

## Mutation closure

| mutation | checker | killed | log sha256 prefix |
| --- | --- | --- | --- |
| `drop_core_safe_boundary` | all_state | True | `c72ba7d02721` |
| `drop_core_target_bank_guard` | all_state | True | `38d498340527` |
| `drop_core_plus_one_guard` | all_state | True | `dc957a7badd1` |
| `drop_core_trust_guard` | all_state | True | `7526fd7f1c47` |
| `allow_active_bank_write` | all_state | True | `ccb25062a676` |
| `accept_bad_crc32` | transition | True | `f131ceca4d92` |
| `drop_commit_cas_expected` | transition | True | `5c4bcb7dadac` |
| `drop_image_version_monotonicity` | transition | True | `d6d1f73ddb06` |
| `drop_both_drain_guards` | transition | True | `d20027192c19` |
| `cancel_keeps_commit_pending` | transition | True | `d7496284ea54` |
| `allow_two_request_conflict` | transition | True | `20bc0b5282b1` |
| `erase_policy_priority_provenance` | all_state | True | `a2a38e81655b` |
| `allow_host_outside_open` | all_state | True | `ed634e207e29` |
| `allow_host_during_policy_pending` | all_state | True | `3716b57662cb` |
| `invert_lkg_rollback_target` | transition | True | `f7edada2abb7` |
| `allow_policy_version_wrap` | all_state | True | `f62779cfd07f` |
| `erase_deadline_fault` | core | True | `0f48df11e4a0` |
| `erase_age_fault` | core | True | `38c3962fd96a` |
| `allow_fallback_action` | core | True | `6d8cf2b3cf89` |
| `erase_registered_output` | core | True | `ab0699df4388` |
| `core_accepts_wrong_activation_version` | core_commit | True | `158e4e9dc8d9` |

## Claim 边界

这是 pre-board、single-mode RTL 证据。板测 latency/power、跨工作 fastest、multimode decoder 已部署到 RTL 仍为关闭状态；T6.25.3 与 T6.25.4 必须在完全相同顶层上分别重跑百万周期 CXXRTL 与三种子 P&R。
