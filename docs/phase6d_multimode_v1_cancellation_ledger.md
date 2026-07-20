# Phase 6D multimode v1 early-stop ledger

## 触发条件

T6.20.4 在 T6.20.3 的 `train` split 上完成 12 个 `d=3` seed-cluster、13 个完整 family、79,872 轮的 paired causal-headroom 实验。最强当前可执行 development baseline 是 static-mixture exact MLD (`p_L=0.111979`)；trusted-bank risk action 最终同为 `0.111979`，relative headroom=`0% [0%,0%]`，没有达到预先写入任务板的 `point >=15%` 且 paired 95% `LCB >=12%`。

完整性证据全部通过：128 个 explicit-coset/official-BSV 样本零 action mismatch，128 个 pure-Julia exhaustive-T-join/official-pymatching 样本零 correction mismatch，512 个 alias-convergence 样本零 action mismatch，future-suffix mutation 在 64-round prefix 内严格不变且 suffix 后真实分叉，15/15 semantic mutations 被拒绝。深审还作废了曾把 generator-only spatial pattern/variance-law loading 传入 decoder 的探索性运行；最终 decoder 只读取过去 folded-residual moments、当前观测和 nominal base sigma，seeds/families/rounds/baselines/gates 未改。因此 early-stop 来源是最终合法候选栈的可用 headroom 不足，不是 runner、数值、因果性或缺失数据失败。

## 状态变更

以下任务按 **T6.20.4 预注册失败分支** 从 `Todo` 改为 `Dropped`，均未执行，也不能被引用为负实验之外的实现证据：

- exact/static backend：`T6.21.1`--`T6.21.4`；
- causal adaptive baselines：`T6.22.1`--`T6.22.4`；
- proposed posterior-predictive/risk-aware stack：`T6.23.1`--`T6.23.5`；
- pilot/formal/SOTA：`T6.24.1`--`T6.24.5`；
- 无合法 teacher 的 learning extension：`T6.26.1`--`T6.26.2`。

T6.20.3 的 calibration、pilot、formal outcome 继续不存在/未访问。T6.20.4 的 explicit d=3 coset 与 T-join 只是正确性/机制 headroom probe，不追认为 T6.21 的完整 source reproduction，也不生成 frozen-benchmark SOTA 结论。

本账本中的“causal ceiling”仅指 T6.20.4 注册的有限 observed-only 诊断候选栈，不是全部因果解码器的数学上确界。NO-GO 关闭的是 Phase 6D v1；新机制只有在全新前瞻 v2 与全新 split 下才可重开。

## 不受影响的路线

`T6.25.1`--`T6.25.4` 的 single-mode production RTL lane 独立继续。它只争取 deterministic 6-cycle、II=1、atomic A/B、CRC/version、LKG rollback、fail-closed、CXXRTL/formal/synthesis/P&R-estimate 贡献；没有真板时仍不得写 measured/fastest。

`T6.26.3`--`T6.26.4` 保留，用于把 multimode 状态明确写成 NO-GO/null，并在 RTL lane 完成后输出 `GO_RTL_ONLY` 或 `NO_GO`，不得做跨 lane 加权补门。

## 重开规则

未来若重开 multimode，必须先提出不同于 v1 的机制假设，并建立新的前瞻 v2 config、全新 development/calibration/pilot/formal split、baseline 与统计门。禁止删除本次不利 family/baseline、调低 15%/12% 门、访问 v1 pilot/formal、把 T6.18.3 的 27.3% 弱-baseline 结果恢复为主张，或用 CNN/RTL 证据补 LER 门。
