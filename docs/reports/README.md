# 人类报告

本目录接收后续任务产生的独立人类报告。任务完成记录仍写入 [`../new_tasks/`](../new_tasks/)，机器证据写入 [`../evidence/`](../evidence/)。

## 目录规则

- 按 `phaseN/` 分层，不为单个 task 新建目录。
- 文件命名为 `<TaskID>_<slug>.md`；标题以 `# <TaskID> <中文标题>` 开始。
- 不使用 `final`、`latest`、`copy` 等无法长期解释的后缀。
- 新报告不得再写入 `docs/` 顶层。

当前已迁移的低绑定报告：

- [`phase4/`](phase4/)：混合慢/快回路、teacher/student 和控制实现报告。
- [`phase6/`](phase6/)：Route-A 与 multimode 分支报告。

未迁移的历史报告仍保留顶层稳定路径，并通过 [`../document_catalog/README.md`](../document_catalog/README.md) 阅读。

