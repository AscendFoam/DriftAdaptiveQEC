# T6.25.1 single-mode RTL 边界与 live-source 审计

## 结论

**`PASS_BOUNDARY_FROZEN_CONVERGED_PRODUCTION_TOP_REQUIRED`**，15/15 gates、20/20 semantic mutations 通过。

当前没有一个 top 同时包含 production CRC32/staging/CAS/drain、Route-A policy/LKG 和 target synthesis surface。旧证据本身没有被否定，但只能作为各自旧 top 的 regression/reference；T6.25.2 必须先形成一个 converged production top，后续 property、百万周期 CXXRTL 和三种子 P&R 必须针对同一 top 重新执行。

## 当前 top 能力矩阵

| module | 实际角色 | 已有能力 | 相对 converged top 缺口 |
| --- | --- | --- | --- |
| `gkp_fast_path_production_top` | production management plus single-mode core | single_mode_map_lut, six_cycle_ii1, event_action_fail_closed, complete_image_crc32, versioned_compare_and_swap, inactive_bank_write_isolation, cancel_drain_snapshot, no_raw_trust_or_cfg_bypass | actual_target_synthesis_surface, lkg_bank_selection, regime_policy_overlay |
| `gkp_fast_path_qualification_top` | long-run raw-pin core qualification wrapper | single_mode_map_lut, six_cycle_ii1, event_action_fail_closed | actual_target_synthesis_surface, cancel_drain_snapshot, complete_image_crc32, inactive_bank_write_isolation, lkg_bank_selection, no_raw_trust_or_cfg_bypass, regime_policy_overlay, versioned_compare_and_swap |
| `route_a_integrated_qualification_top` | single-mode policy plus raw-pin core qualification wrapper | single_mode_map_lut, six_cycle_ii1, event_action_fail_closed, regime_policy_overlay, lkg_bank_selection | actual_target_synthesis_surface, cancel_drain_snapshot, complete_image_crc32, inactive_bank_write_isolation, no_raw_trust_or_cfg_bypass, versioned_compare_and_swap |
| `route_a_hardware_pareto_synth_top` | small-pin synthesis activity harness, no-student profile is primary | single_mode_map_lut, six_cycle_ii1, event_action_fail_closed, regime_policy_overlay, lkg_bank_selection, actual_target_synthesis_surface | cancel_drain_snapshot, complete_image_crc32, inactive_bank_write_isolation, no_raw_trust_or_cfg_bypass, versioned_compare_and_swap |
| `gkp_fast_path_synth_top` | static single-mode core synthesis activity harness | single_mode_map_lut, six_cycle_ii1, event_action_fail_closed, actual_target_synthesis_surface | cancel_drain_snapshot, complete_image_crc32, inactive_bank_write_isolation, lkg_bank_selection, no_raw_trust_or_cfg_bypass, regime_policy_overlay, versioned_compare_and_swap |

关键区别：`gkp_fast_path_production_top` 有完整管理面但没有 policy/LKG；`route_a_integrated_qualification_top` 有 policy/LKG，却直接驱动 core 的 raw `cfg_we` 与 `bank*_trusted`，没有实例化 production management；当前 P&R harness 又包裹后者。因此不能把 T6.2.1、T6.7.3、T6.9.1 的 PASS 横向拼接成同一 actual-top 的 atomic/fail-closed 证明。

## 父证据复用决定

| task | 决定 | 原因 |
| --- | --- | --- |
| T6.2.1 | `REFERENCE_ONLY_NOT_REUSABLE_FOR_CONVERGED_TOP` | production management was exercised for 1,681 cycles without a policy overlay and the legacy report lacks direct source hashes |
| T6.2.2 | `CORE_LONG_RUN_REGRESSION_ONLY` | the million-cycle wrapper drives raw cfg/trust/commit pins and does not instantiate gkp_fast_path_production_top |
| T6.7.3 | `POLICY_CORE_LONG_RUN_REGRESSION_ONLY` | the live policy+core top bypasses production CRC staging and trust ownership |
| T6.9.1 | `OLD_HARNESS_PR_REFERENCE_ONLY` | three-seed P&R is live for route_a_hardware_pareto_synth_top, which wraps the raw-pin qualification top |
| T6.19.1 | `STATIC_CORE_PROFILE_REFERENCE_ONLY` | the only eligible hardware row profiles gkp_fast_path_synth_top, not production management plus policy |

T6.7.3 的 9 个 direct source hashes、T6.9.1 的 12 个 source bindings、T6.19.1 的 13 个 bindings 当前全部 live；T6.2.1/T6.2.2 的旧报告没有直接绑定输入 source hash，已如实保留这个缺口。T6.2.2 的 1,000,000-cycle raw trace hash仍 live，但它只证明 raw-pin core wrapper。

## 与 multimode 软件 lane 的隔离

RTL transitive inventory 中没有 multimode syndrome graph、logical-coset summation、posterior-predictive integration 或 matching 模块。只允许共享四类 transaction contract：candidate image envelope、atomic active view、single-mode regime command、event/action word。接口名相似不构成 multimode decoder 已部署的证据。

## 下一步强制顺序

1. T6.25.2 构建唯一 converged synthesizable production top，并对该 top 完成 property/cover/mutation；
2. T6.25.3 对完全相同的 top 做每 family 至少 100k、aggregate 至少 1M 的 independent-golden/CXXRTL；
3. T6.25.4 对完全相同的 top 做三种子 synthesis/P&R；
4. 真板 latency/jitter/deadline/power 继续为 null，不能声称 fastest。
