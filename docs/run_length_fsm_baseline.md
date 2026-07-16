# T3.2.5 run-length FSM / parameter-bank baseline

## 结论

本任务实现了一个只消费在线可见字段的确定性事件控制器：3-bit 饱和 `e/leakage` run counter、
quadrature phase tie-break、normal/X-recovery/Z-recovery/leakage-hold/fallback 五态，以及真实
`ParamBank` 的双 bank 原子切换。它是 software event-controller baseline，不是 logical decoder、
bit-accurate RTL、综合结果或板测结果。

独立评估的主要结论是负排序但有用：非退化 run-length FSM 相对 static-safe normal 显著降低
synthetic event-control cost，但明显弱于当前 outcome 的 memoryless event controller。原因不是实现失败，
而是 `e_enter_run=2` 对每个短 recovery episode 引入约一拍检测延迟。该结果必须保留，不能把
`run=1` 重新命名为 run-length 来消除差距。

## 在线合同与状态优先级

在线输入只有：当前 q/p residual、observed X/Z `g/e/leakage`、phase bit 和
valid/CRC/fresh/deadline health flags；不含 hidden regime、recovery depth、leakage truth 或 logical truth。

每个周期按以下优先级决定 requested mode：

1. 任一 health flag 失败：`fallback`；
2. observed leakage run 达阈值：`leakage_hold`；
3. 已在 leakage hold 且 clean run 未达 clear threshold：继续 hold；
4. 已在 fallback 且 good-health run 未达 clear threshold：继续 fallback；
5. X/Z `e` run 同时达阈值：用 phase bit 选择本周期 X 或 Z；
6. 单轴 `e` run 达阈值：进入对应 recovery；
7. 否则 `normal`。

默认/生产配置均使用 3-bit counter，最大值为 7，超过后饱和而不回绕。生产训练选择为：

| 参数 | 选择值 |
| --- | ---: |
| `e_enter_run` | 2 |
| `leakage_enter_run` | 1 |
| `leakage_clear_run` | 2 |
| `fallback_clear_run` | 1 |
| correction clip | 1.0 |

`run=1` 单独作为 `memoryless_event` comparator，不进入 run-length 调参网格，避免两条 baseline 退化为
同一个算法。

## 参数银行与失败语义

五种模式各有不可变本地 parameter ROM。requested mode 与 active bank metadata 不一致时，控制器把
完整参数写入 inactive bank 并在同 cycle boundary 原子 commit；没有模式/bank mismatch 时不写 bank。

若外部 slow writer 已占用 pending bank，新写入由真实 `ParamBank` 拒绝。FSM 不消费或覆盖 pending
update，也不只安全一拍：实际状态进入 `fallback`，每拍使用本地 safe ROM；只要 bank 尚不可同步，
后续每拍继续 local-safe。pending update 到期后，FSM 可在同一边界重新写入当前 requested mode 并完成
原子同步。独立冲突探针验证前三拍均 fallback/local-ROM，第四拍同步到 X-recovery，版本从 0 经外部
version 1 到 FSM version 2。

## 调参与评估设计

- training seeds：3 个；evaluation seeds：8 个，严格不重叠；
- 场景：persistent recovery、readout false positive、leakage burst、mixed burst + health fault；
- training base traces：12 条、49,152 cycles；24 组阈值用真实 FSM/ParamBank 重放，共
  1,179,648 FSM cycles；
- evaluation：32 条 unique paired traces、384,000 cycles；
- 同一 trace 比较 `static_safe_normal`、`memoryless_event`、`run_length_fsm` 和不可部署
  `truth_oracle`；
- threshold selection 只用 training truth 计算 evaluator cost，不读取 evaluation truth；在线 FSM 始终
  不见 truth；
- Student-t 区间的 cluster unit 是 8 个 evaluation seed，先在 seed 内平均四场景。

### 事件代价边界

代价矩阵只衡量模式是否匹配 simulator hidden event：正确模式成本为 0；漏 recovery、错轴、漏
leakage/health fault 成本较高；safe fallback 成本小于 unsafe action。每次 bank mode write 另加 `0.002`
抽象成本。该矩阵没有从真实实验时间、脉冲、state disturbance 或 logical lifetime 标定，因此不能称
LER、物理 recovery optimum 或实验成本。

## 生产结果

| Controller | event + write cost | action accuracy | unsafe miss | false intervention | mean event delay (cycle) | mean writes / 12k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static-safe normal | 0.604539 | 0.400034 | 0.878465 | 0 | 2.929986 | 47.91 |
| memoryless event | 0.022917 | 0.955440 | 0 | 0.047494 | 0.032009 | 4379.78 |
| run-length FSM | 0.202829 | 0.793974 | 0 | 0.007888 | 0.981421 | 3126.66 |
| hidden-truth oracle | 0.000628 | 1 | 0 | 0 | 0 | 3768.19 |

Paired seed-cluster comparisons：

- static minus run-length：`0.401710`，95% CI `[0.399014, 0.404407]`；
- memoryless minus run-length：`-0.179911`，95% CI `[-0.180782, -0.179041]`；
- run-length minus truth oracle：`0.202201`，95% CI `[0.201429, 0.202972]`。

run-length 的 false intervention 较 memoryless 低约 6 倍，写 bank 次数也下降，但 detection delay 使
总事件成本更高。所有场景均实际包含 recovery 和 leakage，mixed 场景另包含 1,151 个 health-fallback
truth cycles；正常评估无 bank conflict、所有 correction 有限、active version 精确等于成功写入次数。

## 验证与反简化检查

- 25 项 FSM direct tests：类型/范围失败、3-bit saturation、phase tie、进入/退出迟滞、health priority、
  correction clip、real ParamBank version、copy isolation、replay/gap transactional rejection；
- conflict regression 专门覆盖“第一拍 safe、第二拍错误回旧 bank”的真实缺陷，证明 local-safe 会持续到
  原子重新同步；
- 15/15 machine gates；32 行 Source Data、32 个 unique trace SHA256；
- focused tests：40 passed；术语 registry + focused：54 passed；相邻 runtime/stream/contract：131 passed；
- 显式 `tests/` 全量：`1046 passed, 4 skipped, 4 failed`；四项失败仍只来自 R-N012 登记的
  `fr8_statcalib_extension_lane_benchmark.md` / `P4_benchmark_formal_protocol.md` 缺失；
- 生产 JSON 的 implementation SHA256 绑定 FSM、ParamBank 和 benchmark 源码；术语 registry 另绑定
  `RunLengthParameterBankFSM` 为 software event-controller baseline，并明确不是完整物理闭环 policy。

## Claim 边界

允许写：在当前 protocol-aligned synthetic syndrome stream 中，observed-only run-length event FSM 可用
饱和计数和真实双 bank 原子提交执行，并在明确的事件代价下优于 static-safe、弱于 memoryless。

禁止写：logical-error-rate gain、optimal physical recovery、device-calibrated event cost、bit-accurate RTL、
LUT/FF/BRAM/DSP/Fmax、target-board latency/resource 或真实 GKP 实验结果。
