# Route-A innovation / advantage 原子主张矩阵

本矩阵是 T6.8.7 的论文主张边界，不是把多条 evidence lane 压成一个总分。静态 GKP、一般漂移自适应、Puviani/GQF lifetime 与 FPGA hardware 四类比较保持独立；禁止跨 simulator 相减 lifetime、跨 code family 排 raw latency，hidden-state oracle 也不进入 deployable 排名。

机器可复核版本为 `docs/t6_8_7_route_a_claim_matrix.json`，逐主张 Source Data 为 `docs/t6_8_7_route_a_claim_matrix_source_data.csv`。每条均绑定当前 report/source/config/seed/hash，并为 T6.9.1—T6.9.3 保留显式 `PENDING + null hash`，不能把未来证据当成已完成事实。

## 可进入主线的受限创新点

| 原子主张 | 状态 | 当前最强可支持措辞 | 关键证据 | 不能外推到 |
| --- | --- | --- | --- | --- |
| 合同系统集成 | `SUPPORTED_RESTRICTED` | Route-A 是 contract-centric、regime-aware 的安全编排；锁定 EWMA 的 smooth 主门和百万周期预板 fail-closed correctness 门通过 | T6.7.4 restricted GO；1,000,000 cycles，0 mismatch / undefined / silent overflow | global LER 最优、broad tail 改善、HMM on-FPGA、真实硬件优势 |
| smooth locked-EWMA | `SUPPORTED_PAIRED_OUTCOME` | 预注册 smooth matrix 上 EWMA−Route-A=`2.1687e-5 [1.9003e-5,2.4548e-5]` | 24 seed clusters、20k bootstrap；仅 periodic 通过 Holm family gate | 相对 static/Window 的优势或所有 smooth family 优势 |
| K4 hard-action 等价 | `SUPPORTED_PREBOARD_NARROW` | frozen mean/covariance/prior 的完整 `1024²` syndrome 域内，K4/full static MAP hard action 0 差异 | 1,048,576 点穷举；soft error 约 `1e-14`；retained bits `512/3200` 为代理 | universal exact、综合资源优势或板上时延优势 |
| external BOCD outcome | `SUPPORTED_PAIRED_OUTCOME` | common formal trace 上 Route-A paired LER 低于 pinned external BOCD wrapper | external−Route-A CI=`[7.6679e-6,1.2369e-5]` | matched-budget 优势；该 external 有 1 次 `13,004.1 us > 5,000 us` worst-cap 违例 |
| FPGA deterministic architecture | `SUPPORTED_PREBOARD_NARROW` | integrated Route-A 是经百万周期 bit-exact CXXRTL 资格验证的确定性六周期预板架构 | six cycles；0 mismatch / undefined / silent overflow | ns/Fmax/resource/power、deadline 或真实板上 speed |

## 必须保留的证否、禁止和消融结论

| 原子主张 | 状态 | 当前证据 | 论文处理 |
| --- | --- | --- | --- |
| 相对 static GKP 优势 | `FALSIFIED` | static−Route-A=`-2.4548e-5 [-3.9595e-5,-9.1129e-6]`；average `9.6819e-4 < 9.9274e-4`，且 Route-A worst-window 更差 | static/Window 必须与 proposed 同表；不得选择性只报 Route-A 较低 p95 |
| general drift-adaptive SOTA | `NOT_ESTABLISHED` | budget-qualified external comparator 数为 0；只有一个可执行 external algorithm | 只报“相对 pinned BOCD wrapper 的 paired outcome + budget fail”，不得升级 SOTA |
| 超过 Puviani NMF | `PROHIBITED` | paper-exact 0/15；T6.8.5 八项 prerequisite 全失败，13 个 matched metrics 全为 `null` | 只作 official-source negative reproduction/limitations；reduced probe 不得替代 NMF |
| FPGA speed advantage | `PROHIBITED` | same-task external comparator=0；real-board source-to-action=`PENDING_T6.9.2` | 文献表只作 boundary normalization，不生成 fastest 排名 |
| CNN primary | `ABLATION_ONLY` | legacy checkpoint 未通过 matched schema/budget | CNN 与 teacher/student 仅放消融/扩展证据，主 LER 归因于 MAP，tail safety 归因于 contract/FSM |

## T6.9 解锁规则

- T6.9.1 只能把 integrated P&R 的 Fmax、resource、power 和 latency model 从 `null` 升级为 `estimate`；不能解锁 measured speed。
- T6.9.2 需要相同 bitstream/source hash、至少 `10^6` cycles、零 mismatch/undefined/silent overflow/deadline miss，并分层报告 core/transport/source-to-action/end-to-end 的 p50/p95/p99/worst、II、jitter、resource 与 power。只有 T6.8.6 筛出的同任务可比子集才允许谈 speed advantage。
- T6.9.3 必须逐条重算本矩阵。任何 report/source/config/seed/hash 变化、正向 CI 不再过门、出现非零 RTL/board 错误、T6.9 最终 NO-GO，或措辞越过 lane 边界，均触发对应主张撤销。
- Puviani NMF 主张不由 T6.9 硬件结果解锁；它只在 official exact reproduction 与 same-GQF matched lifetime 两道独立门全部通过后才可能改变。

## 防止“简化实现”的机器门

实现不是静态 Markdown 表：runner 从 T6.7.4、T6.8.1/2/4/5/6 live artifacts 读取数值并计算哈希，生成十条原子主张与 CSV；14 个 gate 分别检查四类对手覆盖、负结果、预算失败、GQF null、FPGA pending、config/seed/hash、T6.9 null 和禁止跨 lane 聚合。14 个 target-specific semantic mutations 逐门尝试隐藏负结果、伪造 hash、发明板测、允许 global score 等，必须全部 fail closed。

验证命令：

```powershell
python -m cnn_fpga.benchmark.route_a_claim_matrix
python -m cnn_fpga.benchmark.route_a_claim_matrix --verify docs/t6_8_7_route_a_claim_matrix.json
python -m pytest tests/test_route_a_claim_matrix.py -q
```
