# T6.20.4 Multimode causal headroom（development-only）

## 结论

**`NO_GO_MULTIMODE_CAUSAL_HEADROOM`**。本任务只使用 T6.20.3 的 `train` split：12 个独立 seed-cluster、13 个完整 family、79,872 轮；未访问 calibration、pilot 或 formal。

最强当前可执行 development baseline 是 `static_mixture_exact_mld`（$p_L=0.111979$），observed-only causal ceiling `risk_aware_observed_only_action` 为 $p_L=0.111979$。相对改善点估计为 **0.00%**，paired seed-cluster bootstrap 95% CI 为 **[0.00%, 0.00%]**，没有达到预注册的 `point >= 15%` 且 `LCB >= 12%`。

这个结果不能写成 SOTA，也不能授权进入 T6.21。它只说明：在当前 train task-signature 下，把 CPD 逐步换成 exact/posterior-predictive backend 的可用因果 headroom 太小，不能支撑预期的 10% formal 优势。

这里的“causal ceiling”只指本任务注册的有限 observed-only 诊断候选栈，不是所有因果解码器上的数学上确界，也不排除使用新机制和全新前瞻 split 的 v2。

## 五段 regret

| 组件 | 替换前 pL | 替换后 pL | 绝对改善 |
| --- | ---: | ---: | ---: |
| estimator | 0.117951 | 0.119228 | -0.001277 |
| metric_likelihood | 0.119228 | 0.152882 | -0.033654 |
| logical_coset_sum | 0.152882 | 0.167518 | -0.014636 |
| posterior_marginalization | 0.167518 | 0.167531 | -0.000013 |
| risk_action | 0.167531 | 0.111979 | +0.055551 |

正数表示降低 LER，负数表示退化。observed-only modewise estimator/likelihood/coset 路径明显退化；trusted-bank robust action 产生 9884 次干预，并相对未保护的 posterior-predictive arm 净减少 4437 个逻辑错误，但最终只回到 strongest static baseline，仍然没有可用 headroom。这是安全回退价值，不是 LER 优势。

## 不删场景的 family 结果

| family | strongest baseline pL | causal ceiling pL | 相对改善 |
| --- | ---: | ---: | ---: |
| burst_outlier | 0.134928 | 0.134928 | +0.00% |
| compound_ood | 0.132487 | 0.132487 | +0.00% |
| correlation_drift | 0.102376 | 0.102376 | +0.00% |
| heavy_tail | 0.125814 | 0.125814 | +0.00% |
| likelihood_mismatch | 0.109049 | 0.109049 | +0.00% |
| mean_drift | 0.104004 | 0.104004 | +0.00% |
| ou_drift | 0.107096 | 0.107096 | +0.00% |
| periodic_drift | 0.109049 | 0.109049 | +0.00% |
| random_walk | 0.101562 | 0.101562 | +0.00% |
| stationary_control | 0.097819 | 0.097819 | +0.00% |
| step_calibration_shift | 0.111979 | 0.111979 | +0.00% |
| telegraph_drift | 0.110840 | 0.110840 | +0.00% |
| variance_drift | 0.108724 | 0.108724 | +0.00% |

## 正确性与反简化检查

- explicit d=3 coset sum 对 official BSV：128 个样本零 action mismatch，最大 log10-odds 误差 2.050e-12。
- 纯 Julia exhaustive T-join 对 official `pymatching`：128 个样本零 correction mismatch；正式长跑不再依赖会崩溃的 PythonCall 路径。
- alias truncation：512 个样本零 action mismatch；概率归一最大误差 4.441e-16。
- future-suffix mutation：prefix action mismatch=0、prefix prior max error=0.0e+00，且 mutated suffix 后 action/posterior 均真实分叉。
- 15/15 个完整性 mutation 被 fail-closed 捕获；所有 13 family 与两个 baseline candidate 均保留。
- 完整性修复账本保留三项被发现并修正的问题；含 generator-only spatial/variance-law privilege 的探索性结果已作废，最终报告没有使用。修复没有改变 seeds、families、rounds、baseline 候选、统计门，也没有在看见结果后调 performance threshold。

## 失败后的约束

Phase 6D v1 不得通过删去不利 family、删除 `static_mixture_exact_mld`/adaptive CPD、改阈值或访问 pilot/formal 来“救”此门。允许的后续是：(1) 将 T6.21--T6.24 标记为本路线 v1 不进入；(2) 继续独立的 single-mode RTL lane；(3) 若未来重开 multimode，必须形成新的、前瞻注册的机制假设，而不是对本数据调参。
