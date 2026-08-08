# Route-A posterior、双影子 bank 与正式入口锁

## 1. 结论

T6.6.3 在 **未创建、未读取 `formal_evaluation`** 的条件下完成 calibration/pilot-only V4 锁：

- machine verdict：`PASS_ROUTE_A_POSTERIOR_AND_COMMON_THRESHOLD_LOCK`，28/28 gates；
- threshold/policy lock：`9347edb270bbeb3f50d8bd8aceaeefd8003e118f1e88712dd5265519bb0f67aa`；
- strongest-baseline selection lock：`55391ca0d426c54fc24dd104295b54ea61c7034e80aa11602a27b6eb74518eee`；
- strongest deployable baseline：`ewma_adaptive_map`；
- V4 deployable experts：`window_map + ewma_adaptive_map`，总更新成本 `1,218/8,192 MAC`。

该结果只允许声称 posterior、event model、baseline、router 与 common tuple 已在 formal 前冻结；不能提前
声称正式 LER/tail 优势、SOTA、RTL/P&R 或实板性能。

## 2. 数据、因果性与 baseline

calibration/pilot 各为 41 cells×12 seeds，分别包含 440,832/566,784 个不重叠 32-decision posterior
updates。五个 domain-separated streams 共导出 4,920 个唯一 seeds；10-bit ADC bin-centre、同一 trace、
2,000-decision parameter period、最近 1,024 decisions 更新等合同对所有 deployable 方法一致。

pilot 五基线各执行 18,137,088 decisions：

| method | equal-family dynamic average LER | p95 | worst | update MAC |
| --- | ---: | ---: | ---: | ---: |
| standard binning | 0.00989787 | 0.0820313 | 0.154297 | 0 |
| static joint MAP | 0.00910600 | 0.0800781 | 0.148438 | 0 |
| Window MAP | 0.00769011 | 0.0703125 | 0.169922 | 128 |
| **EWMA MAP** | **0.00768522** | **0.0703125** | **0.169922** | 136 |
| Kalman MAP | 0.00802924 | 0.0703125 | 0.214844 | 7,121 |

EWMA 仅比 Window 窄胜约 `4.90e-6`，所以它只是 frozen primary comparator，不是 pilot superiority claim。
legacy CNN 因真实 checkpoint 的 schema/budget 不合格留在 ablation；oracle 只作非部署上界。

## 3. posterior 与 common tuple

Route-A HMM class order 固定为 `normal/smooth/calibration_shift/burst`，拒绝旧 leakage/smooth 顺序错位。
选中 `regularization=0.1`、`transition smoothing=4.0`、`temperature=2.0`；pilot
NLL/Brier/accuracy 为 `1.22665/0.47053/0.74700`，smooth recall 仅 `0.36921`，明确不是充分识别证明。

1,728 个 common tuple 全量枚举，最终 candidate 1316 为：

| enter | exit | uncertainty | OOD code | enter hysteresis | recovery hysteresis |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 0.2 | 0.25 | 192 | 2 | 8 |

posterior screening 的最低 abrupt fallback `0.90026`、tail false update `0.01635`、最低 smooth-open
`0.76588`、nominal fallback `0.001519`。557 个 transition 的 mean/p95/max lag 为
`0.167/1/7` posterior windows，零右删失。

## 4. V1--V4 prior-informed 修订与反证

所有修订都发生在 formal 前，且旧反证没有覆盖：

1. **V1 NO-GO**：all-tail event score 伤害 nominal/smooth，无法同时满足 safe false-positive 与 burst
   recall；V2 将 event head 限定为 observed heavy-tail burst evidence。
2. **V2 NO-GO**：38/38 posterior-safe tuples 在完整 pilot LER/catastrophic/nominal selector 中失败；
   tail→初始 static 的最接近 candidate 1316 最小 safety slack 为 `-0.037109375`。机器记录：
   `docs/t6_6_3_v2_static_tail_pilot_selector_nogo.json`。
3. **V3 NO-GO**：真实有状态 freeze-all EWMA 仍 38/38 失败；persistent calibration/telegraph 使 LKG
   变 stale，最小 slack `-0.044921875`。机器记录：
   `docs/t6_6_3_v3_gated_lkg_pilot_selector_nogo.json`。
4. **V4 PASS**：Window/EWMA shadows 持续使用同一 observed window 更新。仅当上一完整窗口的
   pre-update predictive NLL 选择 Window、smooth posterior≥0.30 且 common policy OPEN 时，下一
   2,000-decision period 才 promotion Window；tail/uncertain 强制 validated EWMA shadow。

V4 的 `smooth posterior=0.30` 与 score memory `0.0` 是 pilot-only prior-informed 冻结值；探索中
0.30--0.70 给出相同 aggregate routing。Methods 必须披露这条 revision history，不能写成从研究起点
完全 blind。

## 5. 完整 pilot selector 结果

38/38 posterior-safe tuples 均通过原 T6.5.3 pilot LER/safety constraints，按原排序仍选 candidate 1316：

- 十个 dynamic families 的 EWMA-minus-V4 paired improvement：point `5.1498e-6`，95% CI
  `[3.8783e-6, 6.5486e-6]`；这是 pilot 选择证据，不是 formal claim；
- minimum safety slack `0.0`；step/telegraph/compound 的 average/p95/worst 均与 EWMA 非劣，
  calibration global worst `87/512 = 87/512`；
- periodic pilot average improvement约 `4.90e-5`，mean 仅极小变化，variance/correlation 为 0；优势
  高度集中，T6.7 必须重新检验四-family aggregate；
- scored total fallback `0.50352`，按“fallback 且双方都正确”定义的 overall unnecessary fallback
  `0.49617`；nominal fallback 仅 `0.001628`。高 overall 数字必须作为安全信号代价报告，不能隐藏。

492 条 dual-bank cache 逐 trajectory 重新验证 Window/EWMA 的 32-decision error counts与冻结 baseline
完全一致；最终重跑为 492 hits/0 misses。cache 绑定 trace、truth evaluator、action matrix、calibration、
算法 SHA 与候选集合，只用于断点恢复，不进入选择 core。

## 6. 故障与反简化审计

- prefix causality、truth-key denylist、same-trace intervention 与 truth mutation 均通过；
- commit decision 只读取最后一个 **完整** posterior window，修复了旧 `(stop-1)//32` 最多泄漏
  31 个未来 decisions 的问题；
- late posterior、CRC、version、stale age、event-model hash 五类 runtime mutation fail closed；
- 14 类 semantic mutations 覆盖 formal access、class order、model/lock/baseline/seed/router hash、grid 外
  阈值、CRC、reason、trusted source 等；
- focused posterior/policy/validation suite 为 30/30。

T6.7 只能只读 V4 lock 与 source bindings，在 24 个未触碰 formal seed clusters 上执行。任何 threshold、
event、router、baseline、cadence 或 family weight 修改都必须创建新 protocol；不得回写本锁。
