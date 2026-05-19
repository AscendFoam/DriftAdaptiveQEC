# utils/ — 共享工具库

本目录提供 CNN-FPGA 项目中所有模块共享的基础工具，包括 YAML 配置加载（含继承解析）、路径管理、配置哈希和文件 I/O。

## 文件

| 文件 | 职责 |
|------|------|
| [config.py](config.py) | YAML 配置加载与继承解析、路径管理、JSON/文本文件 I/O |

## 核心 API — `config.py`

### YAML 配置加载

```python
from cnn_fpga.utils.config import load_yaml_config

config = load_yaml_config("cnn_fpga/config/hardware_hil.yaml")
```

`load_yaml_config` 自动处理：

1. 读取 YAML 文件内容
2. 若存在 `base_config` 字段，递归加载父配置
3. 通过深度合并（`_deep_merge`）将子配置覆盖到父配置上
4. 优先使用 `yaml.safe_load`（需 PyYAML），否则使用内置的 YAML 解析器回退

内置 YAML 解析器支持：映射、序列、标量类型（bool, null, int, float, 字符串）、2 空格缩进、行内注释。PyYAML 不可用时的纯 Python 回退方案。

### 配置哈希

```python
from cnn_fpga.utils.config import config_hash

hash_str = config_hash(config)  # 返回 12 字符十六进制哈希
```

用于生成实验运行和数据集的唯一标识符（JSON 序列化 + SHA-256 的前 12 位）。

### 路径管理

```python
from cnn_fpga.utils.config import ensure_dir, get_path, now_tag

# 创建目录（含父目录）
output_dir = ensure_dir("runs/experiment_001")

# 从配置读取路径，带默认值
model_dir = get_path(config, "model_dir", "runs/models")

# 时间戳标签
tag = now_tag()  # "20260519_143022"
```

### 文件 I/O

```python
from cnn_fpga.utils.config import write_text, save_json, open_text

# 原子写入文本文件（先写 .tmp 再 os.replace）
write_text("output/report.txt", "内容")

# 序列化 JSON
save_json("output/summary.json", {"ler": 0.01, "rounds": 10000})

# 打开文本文件（自动创建父目录）
with open_text("output/log.txt", "w") as f:
    f.write("日志内容")
```

`write_text` 使用原子写入模式：先写入 `.tmp` 临时文件，再通过 `os.replace` 替换目标文件，确保写入过程中断时不会损坏已有文件。

### Windows 长路径支持

`_extended_path_str(path)` 在 Windows 上自动添加 `\\?\` 前缀以支持超过 260 字符的路径，非 Windows 系统上为空操作。所有路径操作函数均内置此支持。
