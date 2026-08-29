# T-RISK-20260728-04 V5c resource wall fail-closed 记录

- Task ID：`T-RISK-20260728-04`
- Run ID：`full_v5c_20260827_220849`
- 日期：2026-08-30
- 状态：`Blocked — verified resource wall NO-GO; no scientific execution`

## 输入与不可变边界

- run：`runs/t04_resource_preflight_full_v5c_20260827_220849/`
- config SHA-256：`7e2b507c9d6842c847c82c3398bc3f0a0e0b63d8e3c38bec2c611b1d21cfbafd`
- plan SHA-256：`d2e6fd930c0fb9f0271c0e0ef66e8396fefd1b424b0ca1a2ba01007f6e28376a`
- source snapshot SHA-256：`72b5d90e65d1395123a364d310f8e074e3775f6433699b925efa81d0dbc935cb`
- analysis SHA-256：`719ddcaf0cfc6749b8da9da04e5b6dac303998c0b87b58a649d671933b1632b4`
- 冻结正式分母仍为 518 cells / 2,085,888 rows；本次只执行 outcome-free
  resource profile，没有访问 formal seed 或 formal artifact namespace。

旧 V4、V5、V5b 事务及 staging 全部保持只读，未被 V5c 组合、续跑或投票。

## 终态与 owner 审计

- `owner.lock` 已由 supervisor 正常释放；当前进程树中没有该 run 的 owner 或
  worker。查询命令自身是唯一命中，未启动第二个 supervisor。
- 最终 heartbeat：owner PID `21540`、owner token
  `4f480f2a4643481d99fa1e618f5e99cc`、sequence `6376`、stage
  `inventory_finalize_no_copy`、`profiles_completed=8`、`child_pids=[]`。
- attempt ledger 精确为
  `START_RESOURCE_PREFLIGHT -> FAIL_RESOURCE_PREFLIGHT`；错误为
  `RuntimeError: resource gates failed: wall`。
- `resource_preflight_pass.json` 不存在；终态失败文件为
  `resource_preflight_failed.json`，verdict
  `INCOMPLETE_RESOURCE_FAIL_CLOSED`，不是 scientific NO-GO。
- 当前 boot session 起点为 `2026-08-23T21:07:27.5+08:00`，早于 V5c
  启动；本次不是前两次那样的主机重启中断。

## 完整 raw/resource 证据

- 8/8 full-denominator receipts；observed/expected rows 均为 `227,328`；
  RESET 与 RESET sidecar 均为 `15,360`；exception、missing、conservation
  failure 均为 0。
- inventory raw status 为
  `RAW_EVIDENCE_COMPLETE_NO_SCIENTIFIC_VERDICT`；没有 monolithic archive 或
  merged full CSV；staging 文件数为 0。
- object store 为 78 个唯一对象、`10,860,153,597 B`。
- resource samples 共 37,920 点，active-child samples 37,560，最大 live
  children 精确为 4，peak aggregate RSS=`5,386,903,552 B`。
- stage 观测窗口：formal LPT `0.031--123,295.266 s`，representative
  `123,295.328--190,906.125 s`，joint-maxT
  `190,906.203--191,165.156 s`，inventory
  `191,169.641--191,464.547 s`。

关键文件绑定：

| 文件 | bytes | SHA-256 |
| --- | ---: | --- |
| `heartbeat.json` | 451 | `6451ccea14c3ec779e195196d4b42c3061fef0d318ca6f8cd8cb6f5ca28917a2` |
| `attempts.jsonl` | 1,290 | `13d476dd88c12cbe2c05f392b1e93be4ec8faaeb3d9c6da5ee0b558cd003d377` |
| `resource_samples.jsonl` | 21,577,799 | `629c3b04cfcd66db07b25b126a0a23c2097dd74c98a75254af2c3e7672fe8e73` |
| `inventory.json` | 3,088 | `0d805cfb02efb4344aa2ea46ee4ed9c957af3b5084a9c13ddea8858837a521e0` |
| `resource_preflight_failed.json` | 3,571 | `a955fa06d905f04ee3ada833a286361967b5541fc851934438db7a46ea325f42` |

## wall 失败诊断

从 8 个 receipt 的 commit 时刻、连续 stage 采样、冻结 518-cell plan 和未修改
`stratified_projection` 独立重建得到：

- projected raw four-worker deterministic LPT makespan：约
  `4,273,756.7 s`；
- projected inventory finalize：约 `19,095.4 s`；
- 含 joint-maxT statistics 与 retained-density physicality 的总 wall：约
  `4,293,112.1 s`；
- frozen maximum wall：`1,209,600 s`（14 days）；
- ratio：约 `3.549x`，即约 49.7 days / 14 days；
- projected artifact：约 `135,082,918,009 B`，未触发
  `171,798,691,840 B` artifact gate。

因此 wall 失败不是边界舍入、OOM、GPU 故障、receipt 缺失或外部重启。对同一
源码/合同仅换 run ID 重跑，不能产生诚实 PASS，只会再次消耗约 53 小时完成
profile 后确定性失败。不得通过放宽 14-day gate、删除 cell、缩短 horizon、减少
paired clusters、改 estimand 或减少正式分母救援。

## 实现缺陷与最小修复

V5c 在 gate 前已经计算 `measurements/stats/inventory/projection/decision`，但旧
exception report 只持久化 generic sampling/error，导致终态报告无法直接给出
wall 分解。本次只修复未来失败报告的诊断完整性：

- `phase9_powered_twin_preflight.py` 在 guarded execution 前显式初始化各阶段
  证据；
- `resource_preflight_failed.json` 新增 `completed_stage_evidence`，早期失败保留
  明确 null/empty，晚期 gate 失败保留完整 projection、decision、inventory、
  statistics 与 profile measurements；
- 不修改 V5c 已封存文件，不补写其终态，不改变任一科学门、resource limit、
  seed、分母或 claim boundary；
- 新增 early-worker 与 late-wall 两条回归，focused suite 为
  `25 passed in 74.56 s`。

## 风险复核与插入任务

- `R-N193` 更新为已形成真实 resource terminal：RSS/artifact/inventory 证据完整，
  但 frozen four-worker wall 明确不可行，状态保持 `Open / High / Immediate`。
- `R-N198` 仍为 `Open / Critical / Immediate`：没有 resource PASS，也没有 live
  consumer/V2 seal。
- 新增 `R-N199` 与 `T-RISK-20260830-01`：只允许 outcome-blind、exact-output
  等价的计算/调度可行性修复；任何 fresh full run 之前必须先由独立投影证明
  518-cell/2,085,888-row 合同可在 14 days 内完成。

## 对论文 claim 的影响

本次只产生资源不可行性负证据，不产生 twin、LER、lifetime、physical、hardware
或排名结果。`official_puviani_exact`、`puviani_nmf_surpass`、external SOTA、
round LER、six-state lifetime、physical break-even、hardware measured、rank 与
twin qualification 全部保持 `null`。
