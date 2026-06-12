# T76 Results Callout Sheet

## 1. 用途

本文件把主文 Results / appendix bridge 里最容易直接复用的 callout 句型固定下来。

每条 callout 都包含：

- 对应段落角色；
- 绑定的 `T75-FIG-*` / `T74-*` 资产；
- 当前允许写的口径；
- 当前不能写的口径。

## 2. Callout 清单

| callout_id | 段落角色 | 绑定资产 | 允许写法 | 禁止写法 |
| --- | --- | --- | --- | --- |
| `T76-CALLOUT-R1` | 主文主结果开句 | `T75-FIG-M01`, `T74-TBL-01` | `Under the locked T24 protocol, hybrid_residual_b is the winner across all four frozen scenarios, with ukf as the runner-up.` | `expanded benchmark winner`, `runtime-agnostic superiority`, `board-level winner` |
| `T76-CALLOUT-R2` | 主文主结果补句 | `T75-FIG-M01`, `T74-TBL-01` | `T75-FIG-M01 is a publication-facing compression of T74-TBL-01, which remains the authoritative numeric source.` | `the figure supersedes the table`, `the figure is stronger evidence than T74-TBL-01` |
| `T76-CALLOUT-R3` | 机制/解释层开句 | `T75-FIG-M02`, `T74-TBL-03` | `The six-seed evidence is descriptive: the instability pattern repeats broadly, while the tested lower-clip intervention is mixed and mostly harmful.` | `causal closure`, `the intervention fixes the mechanism`, `teacher necessity proved` |
| `T76-CALLOUT-R4` | 主文结果到附录的 bridge 句 | `T75-FIG-M02`, `T74-TBL-02`, `T74-TBL-03`, `T74-TBL-04` | `Appendix tables provide bounded ablation, cross-seed numeric snapshots, and material provenance that support this conservative reading.` | `appendix upgrades the main result`, `appendix adds stronger deployment validation` |
| `T76-CALLOUT-R5` | 边界/限制段 | `T75-FIG-A01`, `T74-TBL-05`, `T74-TBL-06`, `T74-SUP-03`, `T74-SUP-04` | `Deployment-facing evidence remains layered: isolated true runtime is verified on the current host, whereas the real-board lane remains gate-only and NO_GO.` | `default environment recovered`, `real-board execution success`, `deployment closure` |
| `T76-CALLOUT-R6` | appendix 图注辅助句 | `T75-FIG-A01`, `T74-FIG-04` | `The blocked portability/deployment closure slot remains intentionally unfilled.` | `we can now add a unified closure figure`, `the blocked slot is practically complete` |

## 3. 推荐使用顺序

1. 段落 A 先用 `T76-CALLOUT-R1`，再视版面决定是否接 `T76-CALLOUT-R2`。
2. 段落 B 用 `T76-CALLOUT-R3`，必要时再用 `T76-CALLOUT-R4` bridge 到 appendix。
3. 段落 C 用 `T76-CALLOUT-R5` 收口；若 appendix 直接放 `T75-FIG-A01`，其 caption 或正文引句可配 `T76-CALLOUT-R6`。

## 4. 本文件最重要的限制

- 所有 callout 都必须回到 `T75-FIG-*` / `T74-*` 的既有链路。
- callout 只帮助作者快速落笔，不产生新证据。
- 如果某句需要靠“更强硬件事实”才成立，那它就不应该出现在这里。
