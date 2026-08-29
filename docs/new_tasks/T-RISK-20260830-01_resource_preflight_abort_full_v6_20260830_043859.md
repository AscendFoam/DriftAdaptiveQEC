# T-RISK-20260830-01 V6 source-binding fail-closed

- Run ID：`full_v6_20260830_043859`
- 日期：2026-08-30
- 状态：`Terminal / FAIL_RESOURCE_PREFLIGHT(source binding)`
- run：`runs/t04_resource_preflight_full_v6_20260830_043859/`

## 实际终态

在确认 V5c 无 owner/child、当前 boot=`2026-08-23T21:07:27.5+08:00`、
GPU 无本任务进程后，以提交 `ab4a8e39f791a224780eb544f60b400303343fbf`
启动唯一 fresh V6。supervisor PID=`33104`，owner token=
`244323d308dc46ac980ded8752bfb18d`；四个 formal-LPT child 均在读取 T03
冻结 source binding 时立即拒绝当前 optimized backend-B：
`bound artifact drift: physics/phase9_backend_b.py`。

- attempt：`START_RESOURCE_PREFLIGHT -> FAIL_RESOURCE_PREFLIGHT`；
- `resource_preflight_failed.json` SHA-256：
  `fe18a3014b3d113a4bc9d663023806aeea5dbaad6ee21bb77b8fc226a8725ea3`；
- sample chain：3点、`1,440 B`、SHA-256
  `0b236bd3888d933effe685f7f291b8658aaad43f80620ec541298f38650c07cd`；
- 0 receipt、0 staging、0 object；没有 scientific row、formal seed/outcome 或
  formal artifact namespace 访问；全部 claim 为 literal `null`；
- owner 已释放，supervisor/四 child 均退出；V6 证据永久只读，不得复用 run ID。

## 诊断与后续门

T04 顶层 lineage/source snapshot 已接受新 backend-B，但 worker 会继续通过 T03
released config 的 byte-verified loader；该 config 正确绑定旧 backend-B SHA
`0b7f7e3f...`，所以仅更新 T9.2.3 release pin 不足以释放执行。直接默认注入 source
replacement 会弱化 T03/T04 provenance，自动安全审查已拒绝该方案，相关未验证
代码已全部撤回。

后续必须先获得用户对“建立新的、显式 byte-bound exact-equivalence child contract，
保持旧 T03 config/manifest/read-only raw 不变，并让新 T04 resource run 显式引用该
child”的授权；不得静默放宽旧 source binding、修改旧 T03 证据或再次启动。

## 风险与任务板

`R-N199` 继续 `Critical / Immediate`；`T-RISK-20260830-01` 保持 In Progress，
T04 继续 Blocked。README、风险表和任务板已同步。无需新增科学 task；这是当前
resource repair 的 provenance 子门。

