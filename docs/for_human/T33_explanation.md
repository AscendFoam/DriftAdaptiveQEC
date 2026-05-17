# T33 说明：只清理 Git 跟踪，不删工作区文件

这次 T33 做的是很窄的 repo-hygiene 动作：按照 T19 的清单，把 9 个 `__pycache__` 目录里已经被 Git 跟踪的 `.pyc` 文件从索引里移掉。

我做了三件事：

- 先用 `git ls-files` 复核清单，确认还是那 9 个目录、共 116 个 tracked `.pyc`
- 再执行一次 `git rm --cached -r -- ...`，只对这 9 个目录做 untrack
- 最后验证 `git ls-files | rg "__pycache__|\\.pyc$"` 已经是 0 行，而且 `runs/`、`artifacts/` 没有被碰到

要注意的是，这次是“从 Git 索引移除”，不是“删除工作区文件”。所以 `.pyc` 文件在本地还在，但不会再作为仓库跟踪内容。

这也没有扩大到任何额外清理范围，比如：

- `runs/`
- `artifacts/`
- `.pytest_cache/`
- `.mypy_cache/`
- 源码、配置、benchmark、`.tflite`、硬件路径

所以这次结果可以概括成：tracked cache 已按清单清掉了，但仓库其余历史产物和工作区文件都保持原样。
