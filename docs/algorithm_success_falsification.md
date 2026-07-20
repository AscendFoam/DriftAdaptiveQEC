# T5.1.4 算法成功/证否分支报告

## 结论

T5.1.4 的强学习算法性能分支 **未通过**。当前唯一激活分支为：

`event_aware_adaptive_map_fpga_codesign`

这表示论文方向自动回退为“事件/状态感知、observed-only 的 adaptive periodic MAP 与 FPGA-oriented
parameter-bank/fallback/atomic-update co-design”。它不表示当前已经完成统一 event-aware closed-loop 实验，
也不保留 CNN、TCN、GRU 或其他学习 decoder 的性能主张。

机器 artifact 的 `PASS` 只表示分支规则、反证保留、provenance 和 claim routing 均通过；不表示强分支成功。

## 冻结的强分支必要条件

强分支要求以下八项同时成立：

1. T5.1.1--T5.1.3 与 deployable MAP-LUT safety parents 全部 machine-pass 且 hash-current；
2. 一个明确命名的 learned decoder 候选已在 T5.1.2 decoder lane 执行；
3. 该候选与 static、window、EWMA、Kalman 及适用 strong baseline 消费同一 trace；
4. static Gaussian 下 average `P_L`、window p95 和 observed worst 全部不退化；
5. drift/regime 下对强可部署 baseline 有正的独立 seed-cluster effect；
6. 预注册 family 至少有 Holm-adjusted discovery；
7. 不出现预注册 transient tail violation；
8. 候选只读 observed causal state，并报告可部署范围和成本。

除 parent provenance 外，其余七项均未通过。这里没有用“候选未运行”去推断其性能差，而是将所有未评估
性能门保持为 false；也没有用经典 Kalman 的结果代替 learned candidate。

## 直接证否证据

### 没有 matched learned decoder

T5.1.2 decoder lane 实际执行 `standard/static/window/EWMA/Kalman/oracle`，没有 CNN、TCN、GRU 或其他
学习 decoder。T5.1.1 中的 FNN、RNN teacher 和 distilled student 属于 matched control lane，decision target
是 sBs control residual，不是 syndrome-to-logical-coset decoder，不能跨 lane 借用。

T4.4.5 的 qualified student-retention 结论仍保留在它自己的 two-level finite-cutoff ten-cycle controller
范围内；T5.1.4 只禁止把它写成扩展 decoder matrix 的 learned performance evidence。

### 多重比较未通过

T5.1.3 对 4 个 deployable challengers × 6 scenarios 共 24 个 seed-level hypotheses 做全部 `2^6`
two-sided exact sign flips，再统一 Holm-Bonferroni。正式结果为：

- discoveries：`0`；
- minimum raw p：`0.03125`；
- minimum adjusted p：`0.75`。

因此不能只挑 bootstrap effect CI 或 point estimate 写“可重复优势”，也不能把 1,152 个相关 windows 当成
1,152 个独立 seeds。

### tail 反例必须保留

calibration shift 下，Kalman average `P_L` 和 p95 分别为 `0.005941`、`0.013672`，低于 static 的
`0.021637`、`0.054688`；但 Kalman observed worst 是 `55/512=0.107422`，高于 static 的
`37/512=0.072266`。这说明 average 改善不能自动推出 transient safety。

static Gaussian 下 Kalman 的 average/worst 均低于 static，可作为 fallback 方向的诊断性动机；由于没有
learned candidate 且 Holm family 为 0 discovery，它不是 CNN 性能证据，也不是 universal adaptive superiority。

## fallback 可用证据与限制

fallback 依赖的七项基础合同均当前通过：

- observed-only EWMA/Kalman/sliding-window MAP 已注册；
- window/EWMA/Kalman 已在 decoder lane 同轨执行；
- run-length event controller 与 regime HMM 已注册，但保持 component-only；
- T2.4.3 fixed-point validation；
- T4.2.1 parametric MAP-LUT；
- T4.2.3 conservative fallback；
- T4.3.2 atomic parameter bank。

这些证据只支持后续开发方向和 software/hardware-aware contract。它们尚未形成同一个 finite-energy、
wall-clock matched、end-to-end closed-loop performance matrix，也没有 RTL synthesis、FPGA board、QPU 或
measured hardware 结果。

## 被撤销或隔离的表述

- 不写 CNN/TCN/GRU 超过 static MAP 或 strong adaptive baseline；
- 不写 learned algorithm 在 mixed drift 或全部 10 类噪声上占优；
- 不写 adaptive decoding 普遍优于 static MAP；
- 不把四条 lane-local 模型写成统一端到端鲁棒性实验；
- 不把 T4.4.5 controller teacher/student 或历史 T24 frozen-set 结果升级为 T5.1 decoder confirmation；
- 不把 finite-horizon control reference 写成 global/ten-cycle oracle；
- 不把 fixed-point/LUT software validation 写成 measured FPGA/board/device performance。

## 重开强分支的最低合同

未来只有新 task 在访问新评估数据前预注册独立 seed clusters，并同时满足 matched learned decoder、same
trace、static average/p95/worst nondegradation、strong baseline、positive seed-cluster CI、Holm-adjusted
discovery、无 transient tail violation、observed-only causality 与 deployment cost，才可重开强分支。现有
1,152 windows 禁止改名为独立 seeds。

## 机器产物与验证

- `docs/t5_1_4_algorithm_branch_verdict.json`：16/16 contract gates；
- `docs/t5_1_4_algorithm_branch_verdict_source_data.csv`：278-row parent gate、hash、反证、claim、reopen ledger；
- `tests/test_algorithm_success_falsification.py`：19 个 direct tests；
- mutation tests 拒绝手工强开分支、添加 CNN active claim、篡改 Holm discovery、删除 transient、跨用
  T4.4.5、删除 reopen gate、window 伪重复和硬件 claim 升级；
- stale parent 会使 artifact status FAIL，同时仍不能激活强分支。

