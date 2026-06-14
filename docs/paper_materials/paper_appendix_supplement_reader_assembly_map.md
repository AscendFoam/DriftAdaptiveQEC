# T84 Appendix / Supplement Reader Assembly Map

本表把当前 manuscript route 压成读者化装配图，只回答“放哪一层、怎么写、绝不能写成什么”，不改写既有证据等级。

| surface | recommended_destination | reader_facing_status | boundary_to_keep | next_bounded_action |
| --- | --- | --- | --- | --- |
| 锁定四场景五模式 benchmark 排名与主结果摘要 | `main text` | 已可作为主文核心结果层；当前 note 直接承载这层叙事。 | 只支持 mock-backed software-HIL 排名，不支持 board-level、`.tflite` ranking 或 expanded benchmark。 | `none inside T84; keep frozen` |
| feature / teacher ablation 表与六种子 descriptive 机制材料 | `appendix` | 已可作为“为什么主结果值得这样讨论”的支持解释层。 | 只支持 descriptive support，不支持 causal closure 或 teacher necessity。 | `future bounded mechanism task only if a new causal question is explicitly opened` |
| canonical training/material provenance chain 与 clean CPU-only rerun | `appendix` | 已可作为 provenance/support 表呈现，帮助作者交代材料来源。 | 不能回述成 full training reproducibility closure。 | `future bounded reproducibility task if the paper later needs stronger regeneration evidence` |
| isolated current-host true `.tflite` runtime table for selected preserved artifacts | `appendix` | 已可作为受限 runtime existence 证明。 | 不能升级成 default-env compatibility、software-HIL closure 或 deployment closure。 | `future bounded portability/runtime task on the specific target environment` |
| calibration-extension supplement lane (`statcalib`) | `supplement` | 已可作为单独标记的比较性补充材料。 | 继续保持 no-promotion / no unique clean threshold，不进入 mainline comparator set。 | `future comparator-promotion gate only after broader and cleaner fairness evidence exists` |
| read-only real-board gate / regeneration / provenance pack | `supplement` | 只能以 blocked gate/provenance 形式出现，说明当前 host 仍不能进板级执行。 | 不能写成 board execution success、timing result 或 deployment completion。 | `future host-available real-board smoke/regeneration task` |
| board-level latency / resource / timing rows | `blocked` | 当前主机与仓库事实都不足以支撑这层。 | 需要 `Linux + FPGA` host、openable device path 与真实测量，不能先写 prose 占位。 | `future bounded hardware-execution task on a compatible host` |
| expanded drift families、stronger oracle baselines、new formal benchmark families | `blocked` | 目前不属于 accepted result layer，也不属于本轮 final polish。 | 不能借 final polish 把它们写成“只差排版”的缺口。 | `future bounded benchmark task with a new protocol and fresh evidence pack` |
