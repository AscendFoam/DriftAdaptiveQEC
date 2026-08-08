# T7.3.1：为何不用 exact/oracle MAP？

- verdict：`FAIL_EXACT_ORACLE_MAP_REVIEWER_CONTRACT`
- gates：`18/20`
- semantic mutations：`20/20`

## 可直接用于审稿回复的短答

我们已经把 exact MAP 用作冻结模型下的强 static baseline；没有把它排除。hidden-state oracle 则逐轮读取 simulator 的真实 `theta_t`，其信息不在 deployable observed packet 中，所以它是不可部署的 assumed-model reference，而不是可选择的在线实现。MAP 在已知真实条件分布和 0--1 loss 下给出条件 Bayes 最优，合理目标是用因果可观测信息缩小 static-to-oracle gap，而不是声称超过 oracle。

当前 V4 也不能声称已经缩小该 gap：

- smooth `p_L`：static `0.00096819136`，Route-A `0.00099273964`，oracle `0.00016233656`；
- static-to-oracle gap closure：`-0.03046242`，95% CI `[-0.04966251, -0.01118531]`；
- calibration-shift worst-window：Route-A `181/512`，static `32/512`；
- V5 nested causal selector：`-0.2322%`；新增 action family 只多避免 `9` 个错误，即 `0.02549%`。

因此当前只保留相对预注册 locked EWMA 的窄 paired improvement 与 observed-only/fail-closed 架构主张。未来只有在独立 held-out 漂移上，相对最强 deployable comparator 得到正的 paired gap closure，才可升级为自适应 LER 贡献。若未来真实仪器能在线给出校准状态，该变量应进入新的 observed-input task signature，并计入测量延迟、更新预算和误差。

## 原子证据与边界

| ID | 主题 | 状态 | 来源 | 边界 |
| --- | --- | --- | --- | --- |
| OR-001 | frozen_model_exact_map | `TERM_DEFINITION` | T3.1.2/T6.7.1 | Exact MAP for a specified frozen likelihood is a deployable baseline and is not omitted. |
| OR-002 | hidden_state_oracle | `TERM_DEFINITION` | T1.3.2/T3.1.3 | Per-round MAP receives exact simulator theta_t; it is an assumed-model reference. |
| OR-003 | channel_control_optimum | `TERM_DEFINITION` | T1.4.5/T3.2.9/T5.3.5 | Decoder oracle is not a channel-recovery or finite-horizon control optimum. |
| OR-004 | observed_information | `INFORMATION_BOUNDARY` | T6.5.2 | Deployable methods receive quantized syndrome history, integrity fields, and causal expert state. |
| OR-005 | privileged_information | `INFORMATION_BOUNDARY` | T3.1.3 | Exact mean, covariance, outlier mixture, regime, burst state, and labels remain evaluator-only. |
| OR-006 | online_calibration | `INFORMATION_BOUNDARY` | spitz2018/wagner2021/sivak2024 | Noise estimation uses finite causal data and does not reveal theta_t instantaneously. |
| OR-007 | bayes_risk_role | `DECISION_THEORY` | T1.3.2 | Within the assumed per-round likelihood and zero-one loss, hidden-state MAP minimizes conditional Bayes risk. |
| OR-008 | oracle_nonzero | `DECISION_THEORY` | T3.1.3 | Oracle LER is nonzero (0.02510313); oracle does not mean perfect physical correction. |
| OR-009 | locked_ewma_contrast | `CURRENT_POSITIVE` | T6.7.1 | EWMA minus Route-A is 2.168726038e-05 with a positive paired interval. |
| OR-010 | static_ordering | `CURRENT_NEGATIVE` | T6.7.1 | Smooth static/Route-A/oracle LER = 0.00096819136/0.00099273964/0.00016233656. |
| OR-011 | negative_gap_closure | `CURRENT_NEGATIVE` | T6.7.1 | Static-to-oracle closure = -0.03046242, CI [-0.04966251,-0.01118531]. |
| OR-012 | window_counterexample | `CURRENT_NEGATIVE` | T6.7.1 | Window MAP LER 0.00089641854 is lower than Route-A 0.00099273964. |
| OR-013 | calibration_counterexample | `CURRENT_NEGATIVE` | T6.7.2 | Calibration worst is 181/512 versus static 32/512. |
| OR-014 | causal_selector_stop | `CURRENT_NEGATIVE` | T6.10.1 | Nested selector headroom = -0.2322%; V5 stopped. |
| OR-015 | action_oracle_nonpromotion | `CLAIM_BOUNDARY` | T6.10.1 | Truth-privileged action expansion adds only 9 errors avoided (0.02549%). |
| OR-016 | allowed_current_claim | `CLAIM_BOUNDARY` | T6.8.7/T7.2.3 | Current claim is restricted EWMA improvement plus information-safe architecture, not positive oracle-gap closure. |
| OR-017 | prohibited_claims | `CLAIM_BOUNDARY` | T6.8.7 | Do not claim oracle deployability, oracle superiority, exact-MAP impossibility, or static omission. |
| OR-018 | future_gap_gate | `FUTURE_PROMOTION_GATE` | T7.3.1 | Require held-out positive paired gap closure against the strongest deployable comparator. |
| OR-019 | new_observed_state | `FUTURE_PROMOTION_GATE` | T8.1/T8.2 | A physically measured calibration variable defines a new observed-input signature with delay and budget. |
| OR-020 | population_crossing_audit | `FUTURE_PROMOTION_GATE` | T7.3.1 | Any apparent oracle crossing requires uncertainty, object, model, and implementation audits. |
