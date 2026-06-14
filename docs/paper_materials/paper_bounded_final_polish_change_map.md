# T84 Bounded Final Polish Change Map

本表只记录 `T84` 在 reader-facing final polish 中实际触碰的 section。它不引入新 claim，不改写历史结果，也不提升任何 blocked surface。

## touched_sections

- `Summary of Contributions`
- `Experimental Setup`
- `Numerical Results`
- `Follow-up routes that remain outside the accepted result layer`
- `Discussion`
- `Conclusion`

## Section Map

| section | touched_in_t84 | polish_goal | strongest_supported_truth_retained | untouched_boundary |
| --- | --- | --- | --- | --- |
| `Summary of Contributions` | `yes` | 把 `T24`、`FR8/statcalib`、`current-host NO_GO` 等内部术语压成更接近读者稿的 benchmark / supplement / blocked 语言。 | 六条贡献仍保持原有层级：主结果仍是锁定四场景 software-HIL 排名；机制材料仍是 descriptive support；统计校准仍是补充扩展 lane；runtime/board 仍是分层边界证据。 | 不新增 expanded benchmark、real-board success、default-env `.tflite` portability 或 `statcalib` promotion。 |
| `Experimental Setup` | `yes` | 把主协议改写成“frozen reference benchmark”口径，减少内部任务编号依赖。 | 主实验仍是锁定的 mock-backed software-HIL 对比：四场景、五模式、paired seeds、两次 repeats。 | 不把该协议回述成 board latency、resource、deployment closure 或更广 benchmark。 |
| `Numerical Results` | `yes` | 把结果层组织改写成“主结果 + 支持解释层 + 明确在结果层之外的 follow-up routes”。 | 主文仍只接受冻结参考协议下的五模式排名；ablation 与统计校准仍是边界受限的 supporting / supplement evidence。 | 不把结果层写成 `.tflite` 排名、real-board 排名、expanded benchmark 或成熟 comparator promotion。 |
| `Follow-up routes that remain outside the accepted result layer` | `yes` | 把原先偏内部 closeout/register 的说法收紧成读者可读的 follow-up boundary register。 | Wave A / Wave B 仍只是 sidecar 候选路线，服务未来 bounded task 规划，不属于当前 accepted result layer。 | 不让这些 sidecar 路线改写 frozen anchor，不把统计校准 lane 晋升主线，不把 software-HIL 写成硬件执行。 |
| `Discussion` | `yes` | 压缩内部 route 语言，突出“低维仿射校准合同”与“main text / appendix / supplement / blocked”四层读者路径。 | 最强讨论口径仍是：主线 ranking 成立，机制解释保守，统计校准只是一条 no-promotion 的补充扩展 lane，硬件依赖面仍 blocked。 | 不把 supporting evidence 写成 coequal performance result，不把缺失的 `Linux + FPGA` host 叙述成仅剩格式问题。 |
| `Conclusion` | `yes` | 用更接近作者终稿的语言总结 strongest supported truth，并明确本轮只是 bounded reader-facing polish。 | 结论仍坚持：drift-adaptive affine calibration 是当前最强科学故事；补充层和 blocked surface 继续分层，不升级成完成态。 | 不把 `T84` 写成 submission-ready pack、deployment closure、hardware-ready finalization 或训练/运行链全面闭环。 |
