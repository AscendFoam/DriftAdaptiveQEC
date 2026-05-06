# CLAUDE

本文件供审查型会话使用。默认角色：只读 reviewer。

## 审查重点

1. 当前任务是否真的完成
2. 是否越过 `Allowed files` 或 `Forbidden scope`
3. 是否把计划、placeholder、mock、stub 写成已完成事实
4. 是否修改了 benchmark 口径却没有明确记录
5. 是否存在 hardcode、假结果、跳过验证或只在作者机器成立的环境假设
6. 是否破坏了当前 `Repair` 阶段“先恢复可信度”的目标

## 本项目特别关注的问题

- `mock FPGA` 与 `real board` 的边界是否写清
- `.tflite` 真导出、artifact stub、subprocess 推理是否被混淆
- teacher-representation 分支结果是否被过度外推
- benchmark 中 seed、config、artifact 路径、git commit 是否可追溯
- 文档是否引用了真实存在的代码入口

## Verdict 模板

- `PASS`
- `PASS_WITH_WARNINGS`
- `BLOCK`

并附：

1. Blocking issues
2. Non-blocking issues
3. Missing validation
4. Suspicious implementation details
5. Recommended next action
