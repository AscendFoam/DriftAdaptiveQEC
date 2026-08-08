# Route-A formal experiment preregistration

## 结论

T6.5.3 已在 T6.7 新 formal 结果生成前冻结场景、split、threshold-selection、统计和
GO/NO-GO 规则。协议 verdict 为
`PASS_ROUTE_A_RESULT_BLIND_PREREGISTRATION_FROZEN`，protocol SHA-256 为
`547e66073cbf478352fc180b57a8fff8677a0858df7f4a9aef47f82d88bddd39`。

这是“对新 T6.7 formal 数据 result-blind”的预注册，不是假装项目从未见过先前结果。T5.1.3
的 `55/512 > 37/512`、T5.1.4 的 0 Holm discovery，以及 T5.4.2 的 compound/nominal
反例均在 machine artifact 中披露，并用于把新门槛设得更严格；新 formal seeds、参数组合和
trace 尚未运行。

## 1. 三个互斥 split

| split | seed clusters | rate levels | amplitude levels | duration levels | scored windows/cell |
| --- | ---: | --- | --- | --- | ---: |
| calibration | 12 | 0.0125, 0.025 | 0.08, 0.16 | 16, 32 | 48 |
| pilot validation | 12 | 0.01875, 0.0375 | 0.10, 0.20 | 24, 40 | 64 |
| formal evaluation | 24 | 0.015625, 0.03125, 0.046875 | 0.12, 0.18, 0.24 | 20, 28, 48 | 96 |

seed、transition rate、amplitude、duration 在三 split 间分别两两不相交。calibration 只用于
static image 与非 policy nuisance 参数；pilot 只允许选择一个共同 threshold tuple 和一个最强
observed-only budgeted baseline；formal 不允许 fit、selection、threshold tuning 或 baseline
reselection。

## 2. 场景与 held-out design

smooth family：mean、variance、correlation、periodic；abrupt/OOD family：step calibration
shift、telegraph、burst、readout/reset、leakage、compound；另有 nominal static negative
control。

calibration/pilot 每个动态 family 各 4 个注册 tuple，formal 每个 family 各 6 个平衡 held-out
tuple，总计 143 个 split/family cells。formal workload 为：

- 60 个 dynamic cells + 1 个 nominal cell；
- 24 个真正用于 inference 的 seed clusters；
- 每方法 1,464 trajectories、71,958,528 scored decisions；
- 七个 deployable candidates 合计 503,709,696 decisions；
- oracle 的 71,958,528 decisions 独立分栏，不进入 deployable aggregate。

amplitude 不是跨 family 偷换的裸数字：每个 family 都冻结了物理/有效模型映射。例如 mean
使用 lattice fraction、variance 使用 sigma ratio、correlation 使用 rho、burst 使用 outlier
probability/scale、readout/reset 使用 confusion/dwell、leakage 使用 injection/persistence。

## 3. 共享 trace 与因果位置

所有方法回放同一条 immutable trace 和 action opportunities。latent dynamics、syndrome、
readout/reset、leakage、transport 使用 SHA-256 domain-separated seed streams。每个 cell 先有
8 个共同未计分 nominal preamble；smooth 从首个 scored window 开始，abrupt onset 用 seed/cell
hash 固定在 scored offset 0—7。任何 event timing 都不能依赖方法输出。

## 4. 阈值如何冻结

T6.5.3 只冻结候选网格和唯一选择规则，不提前填写 desired threshold：

- regime enter posterior：0.60/0.70/0.80/0.90；
- regime exit posterior：0.20/0.30/0.40；
- uncertainty fallback：0.25/0.35/0.45/0.55；
- OOD code：128/160/192/224；
- enter hysteresis：2/3/4 windows；recovery：4/6/8 windows。

T6.6.3 只能在 pilot validation 上按以下固定顺序选择一组供所有场景共享的 tuple：

1. integrity/undefined/silent-overflow 为零；
2. 全部 pilot abrupt/OOD catastrophic 与 nominal non-inferiority 通过；
3. 最大化最差 family safety slack；
4. 最大化 aggregate paired LER improvement 的 95% LCB；
5. 最小化 unnecessary fallback，再最小化 fallback；
6. 仍并列时按固定 tuple 字典序。

若没有 tuple 满足全部约束，则 formal 为 NO-GO；不能看 formal 后回头选阈。选定 tuple、strongest
baseline 与 parent hashes 必须写入新的 T6.6.3 lock SHA，而不是覆盖本预注册。

## 5. 统计合同

- 独立 cluster 是 seed；同一 seed 的所有 families/cells/windows 一起 bootstrap；
- 20,000 次 paired nonparametric cluster bootstrap，seed `202607176999`，95% bounds；
- 汇总顺序固定为 decision -> cell -> family -> seed；smooth 四 family 等权，abrupt/OOD 六
  family 等权，禁止 raw decision pooling 或事后 reweighting；
- strongest baseline 只在 pilot 的十个 dynamic family 等权 aggregate 上选，按 average LER、
  p95、worst、cost、canonical id 排序；oracle 永久排除；
- smooth family superiority 与 abrupt/OOD safety 分别做 Holm step-down，FWER 0.05；
- `p_X,p_Y,p_Z` 和 cost 为 secondary descriptive intervals，不单独激活主 claim。

每个 tail window 是不重叠 512 decisions；p95 用 empirical `higher` quantile；同时报告 seed-level
worst 和 global worst 的 `numerator/512 + trace locator`，不能把 windows 当独立 seeds。

## 6. 预注册通过门

### Average/smooth 主门

相对 pilot 冻结的最强 deployable baseline，`baseline LER - Route-A LER` 的 aggregate smooth
paired 95% LCB 必须严格 `>0`。

### Calibration-shift tail

- Route-A global worst error count 不得高于 baseline；
- seed-worst paired difference 的 95% UCB 必须 `<=0`；
- 因此新独立 run 不允许重现 Route-A `55/512 > 37/512` 这一方向的反例。

### 每个 abrupt/OOD family 的 catastrophic gate

所有条件同时满足：average LER difference 95% UCB `<=0.002`；p95 difference UCB
`<=4/512`；seed-worst difference UCB `<=8/512`；任何单 window excess 不超过 16 errors。

### Nominal non-inferiority

所有条件同时满足：average LER difference UCB `<=0.0005`；fallback rate `<=1%`；
unnecessary fallback `<=0.75%`；induced-minus-avoided rate UCB `<=0.00025`。

### 失败分支

- tail/catastrophic 失败：只保留 smooth-only；
- average 失败：回退 static MAP-LUT + deterministic FPGA；
- CNN checkpoint/budget/performance 失败：只进消融/supplement；
- 真板缺失/失败：只写 hardware-aware/CXXRTL/P&R estimate。

## 7. 反简化审计

23 个 gate 全部独立重算，12 类 semantic mutation 全部拒绝，包括 seed/rate overlap、
per-scenario thresholds、formal tuning、oracle baseline、window pseudoreplication、低 bootstrap、
放宽 average/tail/nominal 门和原地覆盖 protocol。

深审后额外冻结了等权聚合和 SHA-256 trace/onset 规则；否则即使 split 表面互斥，仍可通过
post-result weighting 或方法依赖事件位置改变结论。

## 8. 当前证据边界

当前只允许声称“新 formal 实验的 result-blind protocol 已冻结”。不得声称 threshold 已选、
strongest baseline 已选、formal 已执行，或 Route-A 已取得 average/tail/safety 优势。

