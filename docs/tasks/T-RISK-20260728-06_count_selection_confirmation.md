# T-RISK-20260728-06：有限 count 选择与确认（legacy mirror）

- **日期**：2026-07-30
- **状态**：Done（Independent PASS）
- **权威完成记录**：
  `docs/new_tasks/T-RISK-20260728-06_count_selection_confirmation.md`

## 输入与实际完成

本任务在 T04 outcome不可见时冻结 scale `{1.5,2.0,2.5,3.0}`、完整3,043门、
互斥 selection/confirmation seeds、最小通过规则与候选耗尽NO-GO。
四个候选全部执行后，选择最小passing scale=`2.0`，即
state/round/fault count=`768/1536/4608`。

## 产物与验证

- selection/confirmation=`8+24` chunks、`1,024+6,144` density trials；
- selection maxT=`796` replicates/`160` power cases；
- confirmation maxT=`199` replicates/`40` power cases；
- 4个 linked blueprints，最终3,043门=`3,037 stochastic + 6 exact`；
- physics-free verifier=`21/21`，max delta=`1.94e-16`；
- verification analysis=
  `5a49967fddc86283a42ff75091ce099eed9d3ae799763a3d1e4cca40c7a19c2e`。

## 风险、插入任务与任务板同步

- R-N185、R-N192、R-N194 降为 Mitigated；R-N193继续Open；
- 不插入新任务，按顺序启动 T-RISK-20260728-04；
- 本PASS只资格化T04 count/preregistration，不是twin、LER、lifetime、
  physical、hardware、official Puviani或external SOTA证据；
- `docs/new_task_board.md` 中本任务为 Done (Independent PASS)。
