# Repository Hygiene Evidence Packs

本目录保存仓库噪声、tracked cache、cleanup manifest 相关文档。

## 文件清单

| 文件 | 来源任务 | 用途 |
| --- | --- | --- |
| `cleanup_tracked_cache_manifest.md` | `T19/T33` | tracked `__pycache__` / `.pyc` inventory and bounded cleanup manifest |

## 边界

`cleanup_tracked_cache_manifest.md` 只覆盖已跟踪缓存/字节码文件。它不覆盖 `runs/`、`artifacts/`、源码、配置或历史实验结果清理。
