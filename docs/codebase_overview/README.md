# Codebase Overview Documents

> **状态提示：** 本目录是代码阅读辅助，不是当前任务状态源。当前入口见 [`../README.md`](../README.md)。

本目录统一存放代码库阅读辅助文档，合并来源为：

- `docs/intro/`
- `docs/stucture_explanation/`

原两个目录已退役。后续如果要补充 `physics/`、`cnn_fpga/`、配置、运行入口或调用关系说明，统一放在本目录。

## 文件清单

| 文件 | 内容 | 使用方式 |
| --- | --- | --- |
| `physics_library.md` | `physics/` 物理仿真库说明、API 草图、数学背景、旧代码审查记录 | 用于理解物理层模块职责；不直接作为当前实验事实来源 |
| `physics_directory_structure.md` | 早期 `physics/` 目录功能、调用关系和逐文件说明 | 用于理解物理层代码结构；具体文件列表和状态需按当前仓库复核 |
| `cnn_fpga_directory_structure.md` | `cnn_fpga/` 目录职责、静态依赖、入口脚本、配置字段与 I/O 产物说明 | 用于读代码和定位入口；具体文件列表和命令需按当前仓库复核 |

## 边界说明

1. 本目录只负责“代码怎么读、模块怎么连”的说明，不承担当前计划、当前任务或当前证据状态。
2. 当前项目事实与任务入口以 `docs/README.md` 的导航为准。
3. 本目录中的旧命令、旧路径、旧完成度判断和旧 review 记录只能作为历史阅读辅助；涉及 `.tflite`、real-board、P4 benchmark、statcalib 或论文 claim 时，必须回到对应 task/review/run/artifact 核对。
4. 如果后续重写这些长文档，应优先补“当前文件清单与入口索引”，不要把它们扩写成新的实验计划入口。
