# T85 Residual State-Lag Sweep

本表只记录 `T85` 在当前主线 note 中发现并处理的 residual wording-lag。它不改写实验事实，只清理“已经做完的工作仍被写成未来待做”这类状态滞后。

## screened_scope

- `Summary of Contributions`
- `Experimental Setup`
- `Numerical Results`
- `Discussion`
- `Conclusion`

## touched_locations

| location | stale_wording_summary | action_taken | boundary_preserved |
| --- | --- | --- | --- |
| `Discussion` | `remaining manuscript-side work is reader-facing condensation and route cleanup` 仍把 `T84` 已完成的 condensation / route cleanup 写成后续待执行事项。 | 改写为“剩余 manuscript-side 工作是基于现有分层做 bounded submission-facing assembly”，把下一步从“再做 final polish”收紧为“在既有边界上做装配”。 | 未新增 claim；未把 blocked surface 升格；仍明确不是新证据层 promotion。 |
| `Conclusion` | `remaining writing work is to translate these internal layers into a final reader-facing polish pass` 与 `bounded reader-facing polish` 两处措辞，仍把 `T84` 已完成的 reader-facing final polish 写成未来工作。 | 改写为“在已翻译好的分层基础上做 bounded submission-facing assembly”，并明确它不等于 deployment closure、submission-ready completion 或 hardware-ready finalization。 | 保留 simulation/material-first 路径；保留 `.tflite`、real-board、expanded benchmark、full reproducibility 等 blocked/support-only 边界。 |

## screened_but_untouched

- `Summary of Contributions`：本轮未发现新的 state-lag wording；维持 `T84` reader-facing 翻译结果。
- `Experimental Setup`：本轮未发现新的 state-lag wording；维持 frozen reference benchmark 口径。
- `Numerical Results`：本轮未发现新的 state-lag wording；维持“主结果层 / supporting interpretation / results-layer-external follow-up routes”三层结构。
