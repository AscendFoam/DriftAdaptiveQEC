"""Configuration helpers for YAML-driven experiment scripts."""

# 中文说明：
# - 提供统一配置读取接口（优先 PyYAML，缺失时回退到内置解析器）。
# - 同时提供路径创建、配置哈希和 JSON 落盘等基础能力。

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from ast import literal_eval
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config from disk."""
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    data = _load_yaml_file(cfg_path)
    base_ref = data.get("base_config")
    if not base_ref:
        return data
    base_cfg = load_yaml_config((cfg_path.parent / str(base_ref)).resolve())
    return _deep_merge(base_cfg, {key: value for key, value in data.items() if key != "base_config"})


def _load_yaml_file(cfg_path: Path) -> Dict[str, Any]:
    """Load one YAML file without resolving `base_config`."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        # 中文注释：运行环境没有 PyYAML 时，使用本文件末尾的轻量解析器兜底。
        with cfg_path.open("r", encoding="utf-8") as f:
            content = f.read()
        data = _load_yaml_fallback(content)
    else:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {cfg_path}")
    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def config_hash(config: Dict[str, Any], length: int = 12) -> str:
    """Return a short stable hash for a config dict."""
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def now_tag() -> str:
    """Timestamp for run folder naming."""
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _extended_path_str(path: str | Path) -> str:
    """Return a Windows long-path-safe string while remaining a no-op elsewhere."""
    resolved = Path(path).expanduser().resolve()
    raw = str(resolved)
    if os.name != "nt":
        return raw
    if raw.startswith("\\\\?\\"):
        return raw
    if raw.startswith("\\\\"):
        return "\\\\?\\UNC\\" + raw.lstrip("\\")
    return "\\\\?\\" + raw


def ensure_dir(path: str | Path) -> Path:
    """Create directory when missing and return resolved Path."""
    p = Path(path).expanduser().resolve()
    os.makedirs(_extended_path_str(p), exist_ok=True)
    return p


def get_path(config: Dict[str, Any], key: str, default: str) -> Path:
    """Read a path key from config.paths."""
    paths = config.get("paths", {})
    raw = paths.get(key, default)
    return Path(raw).expanduser().resolve()


def open_text(path: str | Path, mode: str, *, encoding: str = "utf-8", newline: str | None = None):
    """Open a text file with parent directory creation and Windows long-path support."""
    out_path = Path(path).expanduser().resolve()
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        ensure_dir(out_path.parent)
    return open(_extended_path_str(out_path), mode, encoding=encoding, newline=newline)


def write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically write UTF-8 text content."""
    out_path = Path(path).expanduser().resolve()
    ensure_dir(out_path.parent)
    temp_path = out_path.with_name(f"{out_path.name}.tmp")
    try:
        with open(_extended_path_str(temp_path), "w", encoding=encoding, newline="") as f:
            f.write(content)
        os.replace(_extended_path_str(temp_path), _extended_path_str(out_path))
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write JSON with UTF-8 and indentation."""
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _strip_comment(line: str) -> str:
    if "#" not in line:
        return line
    # Keep this simple parser deterministic for config templates.
    return line.split("#", 1)[0].rstrip()


def _to_tokens(content: str) -> List[Tuple[int, str]]:
    tokens: List[Tuple[int, str]] = []
    for raw in content.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"YAML fallback parser requires 2-space indentation: {raw!r}")
        tokens.append((indent, line.strip()))
    return tokens


def _parse_scalar(text: str) -> Any:
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "none", "~"):
        return None

    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]

    if text.startswith("[") and text.endswith("]"):
        try:
            return literal_eval(text)
        except Exception:
            inner = text[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(item.strip()) for item in inner.split(",")]

    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", text):
        return float(text)
    return text


def _parse_block(tokens: List[Tuple[int, str]], idx: int, indent: int):
    if idx >= len(tokens):
        return {}, idx

    first_indent, first_content = tokens[idx]
    if first_indent != indent:
        raise ValueError(f"Unexpected indentation at token {idx}: {tokens[idx]}")

    if first_content.startswith("- "):
        # 中文注释：解析 YAML 列表块（例如 scenarios: 下的多个场景）。
        out: List[Any] = []
        while idx < len(tokens):
            cur_indent, cur_content = tokens[idx]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise ValueError(f"Invalid list indentation near token {idx}: {tokens[idx]}")
            if not cur_content.startswith("- "):
                break

            rest = cur_content[2:].strip()
            idx += 1
            if not rest:
                if idx < len(tokens) and tokens[idx][0] > indent:
                    child, idx = _parse_block(tokens, idx, tokens[idx][0])
                    out.append(child)
                else:
                    out.append(None)
                continue

            if ":" in rest and not rest.startswith("["):
                key, val = rest.split(":", 1)
                key = key.strip()
                val = val.strip()
                item: Dict[str, Any] = {key: _parse_scalar(val) if val else None}
                if idx < len(tokens) and tokens[idx][0] > indent:
                    child, idx = _parse_block(tokens, idx, tokens[idx][0])
                    if isinstance(child, dict):
                        item.update(child)
                    else:
                        item["_items"] = child
                out.append(item)
            else:
                out.append(_parse_scalar(rest))
        return out, idx

    # 中文注释：解析 YAML 映射块（key: value）。
    out_dict: Dict[str, Any] = {}
    while idx < len(tokens):
        cur_indent, cur_content = tokens[idx]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise ValueError(f"Invalid mapping indentation near token {idx}: {tokens[idx]}")
        if cur_content.startswith("- "):
            break

        if ":" not in cur_content:
            raise ValueError(f"Expected key:value near token {idx}: {tokens[idx]}")
        key, val = cur_content.split(":", 1)
        key = key.strip()
        val = val.strip()
        idx += 1
        if val:
            out_dict[key] = _parse_scalar(val)
        else:
            if idx < len(tokens) and tokens[idx][0] > indent:
                child, idx = _parse_block(tokens, idx, tokens[idx][0])
                out_dict[key] = child
            else:
                out_dict[key] = None
    return out_dict, idx


def _load_yaml_fallback(content: str) -> Dict[str, Any]:
    tokens = _to_tokens(content)
    if not tokens:
        return {}
    parsed, idx = _parse_block(tokens, idx=0, indent=tokens[0][0])
    if idx != len(tokens):
        raise ValueError("YAML fallback parser did not consume all tokens.")
    if not isinstance(parsed, dict):
        raise ValueError("YAML fallback parser expects mapping root.")
    return parsed

"""
### 代码整体功能解析
这段代码是一个**轻量级的 YAML 配置文件处理工具**，专为实验脚本设计，核心目标是提供统一、可靠的配置读取能力，并附带实验管理常用的辅助功能（路径管理、配置哈希、JSON 保存等）。

---

### 核心模块逐行解析
#### 1. 基础导入与类型定义
```python
from __future__ import annotations
import datetime as dt
import hashlib
import json
import re
from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, List, Tuple
```
- 导入了文件路径处理（`Path`）、哈希计算（`hashlib`）、JSON 序列化（`json`）、日期时间（`datetime`）等基础库；
- `literal_eval` 用于安全解析字符串形式的 Python 字面量（如列表、数字）；
- 类型注解（`Dict[str, Any]` 等）提升代码可读性和类型检查能力。

#### 2. 核心函数：加载 YAML 配置 (`load_yaml_config`)
```python
def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()  # 标准化路径（解析 ~、绝对路径）
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    try:
        import yaml  # 优先使用 PyYAML 库
    except ImportError:
        # 无 PyYAML 时，用内置轻量解析器兜底
        with cfg_path.open("r", encoding="utf-8") as f:
            data = _load_yaml_fallback(f.read())
    else:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)  # PyYAML 安全加载（避免代码注入）
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {cfg_path}")
    return data
```
- **核心逻辑**：优先使用成熟的 `PyYAML` 库解析配置，若环境缺失则自动切换到内置的极简 YAML 解析器；
- **安全性**：使用 `yaml.safe_load` 而非 `yaml.load`，避免恶意 YAML 中的代码执行风险；
- **健壮性**：校验文件存在性、配置根节点必须是字典（避免非键值对格式的 YAML）。

#### 3. 辅助工具函数
| 函数名 | 功能 | 核心逻辑 |
|--------|------|----------|
| `config_hash` | 生成配置的短哈希值 | 将配置字典 JSON 序列化（排序键保证稳定性）→ SHA256 哈希 → 截取前 N 位，用于实验版本标识/去重 |
| `now_tag` | 生成时间戳字符串 | 格式 `%Y%m%d_%H%M%S`（如 20260316_153045），用于实验文件夹命名 |
| `ensure_dir` | 确保目录存在 | 递归创建目录（`parents=True`），忽略已存在（`exist_ok=True`），返回标准化路径 |
| `get_path` | 读取配置中的路径 | 从 `config["paths"][key]` 读取路径，支持默认值，自动标准化路径 |
| `save_json` | 保存 JSON 文件 | 支持 UTF-8 编码、缩进格式化，自动创建父目录 |

#### 4. 内置 YAML 兜底解析器（核心私有函数）
当环境没有 `PyYAML` 时，通过纯 Python 实现极简 YAML 解析，核心分为 4 步：
1. `_strip_comment`：移除 YAML 行内注释（`#` 后的内容）；
2. `_to_tokens`：将 YAML 内容拆分为「缩进级别 + 内容」的令牌列表，强制 2 空格缩进；
3. `_parse_scalar`：解析基础类型（布尔、空值、字符串、数字、列表）；
4. `_parse_block`：递归解析嵌套的映射（`key: value`）和列表（`- item`）；
5. `_load_yaml_fallback`：整合上述步骤，完成最终解析并校验结果。

⚠️ 注意：兜底解析器仅支持 YAML 核心语法（键值对、列表、基础类型），不支持锚点、复杂数据类型等高级特性。

---

### 总结
1. **核心能力**：提供「PyYAML 优先 + 内置解析器兜底」的 YAML 配置读取，保证环境兼容性；
2. **辅助能力**：封装实验脚本常用的路径创建、配置哈希（实验标识）、时间戳生成、JSON 保存等功能；
3. **设计特点**：轻量、健壮（含文件/格式校验）、兼容（无第三方依赖时可用），专为 YAML 驱动的实验脚本场景优化。
"""
