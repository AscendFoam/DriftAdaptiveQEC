# T6.15.5：Route-A V5 simulation/pre-board 最终 evidence gate

## 终态

Phase 6B 走预注册 early-stop 路径，终态为 `NO_GO_V5_EARLY_HEADROOM_STOP`。这表示流程正确完成，但当前 V5 方法没有进入实现、pilot、formal 或硬件资格验证；不是软件异常，也不是可由补 seed 挽救的“不显著”。

T6.10.1 的机器结果被独立重算：strict-causal selector 为 35,396 errors，outer-training-selected strongest baseline 为 35,314 errors，headroom=`-0.2322%`；hard-decision oracle 与 expanded posterior candidate oracle 分别为 29,105/29,096 errors，纯 incremental action-space headroom=`0.02549%`。二者分别低于 `10%/12%` 入口门。

## 防止错误升级

- expanded candidate oracle 的 overall truth-privileged headroom=`17.6077%` 不参与 action-space gate；它几乎全部来自不可部署的 per-decision expert selection。
- T6.10.2—T6.15.4 共 20 个条件 task 均为 `Dropped`，不是 `Done`；没有创建 V5 contract、四分割、formal manifest/output、quantized action、CXXRTL、formal property 或 P&R profile。
- V4 formal 只在 T6.10.1 作 diagnostic replay；不得改名为 V5 formal。
- T6.9.2 仍有 42 个 measured fields 为 null；V4 完整论文仍保持 T6.9.3 的 NO-GO。
- Phase 6C 仅以 `READ_ONLY_AUXILIARY_COMPARISONS` 模式开放，不能修改 Phase 6B verdict、baseline、门槛或 claim。

## claim registry

十条原子 claim 分为：三条性能主张 `REVOKED/NOT_RUN_EARLY_STOP`，五条 V5 formal/quantized/RTL/P&R 主张 `NOT_RUN_EARLY_STOP`，measured hardware 为 `BLOCKED`，Phase 6C 只有只读许可。不存在 “missing=pass” 或 overall verdict 覆盖原子负结果。

## 验证

- 12/12 evidence gates；
- 6/6 target-specific semantic mutations；
- 20/20 conditional task status exact；
- 0 个 V5 downstream output；
- 10-row Source Data；
- focused tests：5 passed；
- repository validator：Source Data、parent hash、Dropped status、absence、gate 与 analysis hash 全通过。

机器报告：`docs/t6_15_5_route_a_v5_final_evidence_gate.json`；Source Data：`docs/t6_15_5_route_a_v5_final_evidence_gate_source_data.csv`。
