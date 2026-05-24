# Ollama 上的 GLM-OCR 代码调用

本文档描述本项目当前已经跑通的本地调用方式：在 Windows 本机安装 `Ollama`，拉取 `glm-ocr:latest`，然后通过 HTTP 调用 `http://localhost:11434/api/generate` 完成 OCR。

适用场景：

- 普通 PDF
- 截图
- 票据
- 通用 OCR 任务

不适合直接照搬到古籍竖排主链路的部分：

- 古籍项目主链路仍然优先使用 `CHAT_models + Kraken`
- `GLM-OCR Local` 在本项目里定位为第二本地 OCR 引擎

## 1. 前置条件

本机需要满足：

```powershell
ollama list
```

能看到：

```text
glm-ocr:latest
```

本机服务地址默认是：

```text
http://localhost:11434
```

探测服务是否正常：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags"
```

## 2. 调用接口

本项目当前使用的是：

```text
POST /api/generate
```

完整地址：

```text
http://localhost:11434/api/generate
```

请求体关键字段：

```json
{
  "model": "glm-ocr:latest",
  "prompt": "Read the document image carefully and return only the OCR text. Preserve original language and line breaks.",
  "stream": false,
  "images": ["<base64_png_or_jpg>"]
}
```

返回结果里本项目目前主要使用：

- `response`
- `done`
- `done_reason`

`response` 就是 OCR 文本。

## 3. 最小 Python 调用示例

下面是最小可跑版本，不依赖本项目其它模块：

```python
import base64
import json
from pathlib import Path
from urllib import request


def glm_ocr_local(image_path: str | Path) -> dict:
    image_path = Path(image_path)
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    payload = {
        "model": "glm-ocr:latest",
        "prompt": "Read the document image carefully and return only the OCR text. Preserve original language and line breaks.",
        "stream": False,
        "images": [image_b64],
    }

    req = request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


result = glm_ocr_local("test.png")
print(result["response"])
```

## 4. 本项目中的实际封装

本项目已经把本地调用封装在：

- [salt_e4_glm_local.py](D:/Codes/generic/legal_understanding/salt_e4_glm_local.py)

核心入口有两个：

1. `detect_local_glm_status(profile)`
2. `ocr_image_local(profile, image_path)`

### 4.1 状态探测

`detect_local_glm_status(profile)` 会检查：

- Ollama 服务是否能访问
- `glm-ocr:latest` 是否已拉取

返回值示例：

```python
{
    "available": True,
    "provider": "glm_ocr_local",
    "mode": "ollama",
    "host": "http://localhost:11434",
    "api_path": "/api/generate",
    "model": "glm-ocr:latest",
    "issues": [],
    "models": ["glm-ocr:latest"]
}
```

### 4.2 OCR 调用

项目当前实际调用逻辑等价于：

```python
from pathlib import Path
from document_profiles import load_profile
from salt_e4_glm_local import ocr_image_local


profile = load_profile("generic_vertical")
result = ocr_image_local(profile, Path("test.png"))
print(result["response"])
```

## 5. 为什么项目里要先降采样

这是当前本机实测后的必要处理。

直接把很高的长条裁图发给 `glm-ocr:latest`，Ollama 端可能返回：

```text
HTTP 500
GGML_ASSERT(...)
```

本项目已经在 `salt_e4_glm_local.py` 里加了自动降采样：

- 如果图片最长边大于 `1024`
- 先缩小到最长边 `1024`
- 再转 base64 调用模型

这一步对应函数：

- `_downsample_pixmap_bytes(image_path, max_side=1024)`

这不是通用理论建议，而是当前机器上已经验证过的稳定性修正。

## 6. 项目里的整页兜底

`GLM-OCR Local` 在本项目中不再强依赖 E3 列切分。

当前逻辑是：

- 如果已有 `layout_json_e3/page_xxx.json`
  - 按列裁图 OCR
- 如果没有 E3 布局
  - 自动退回 `whole_page_fallback`
  - 整页渲染后直接 OCR

对应函数：

- `prepare_page_crops(...)`

这意味着通用文档场景下，即使没有古籍那套布局前处理，也能直接调用。

## 7. curl 示例

如果你想直接用命令行调接口，可以这样做：

先把图片转成 base64，再拼 JSON 请求体。PowerShell 示例：

```powershell
$bytes = [System.IO.File]::ReadAllBytes("D:\test.png")
$b64 = [Convert]::ToBase64String($bytes)
$body = @{
  model  = "glm-ocr:latest"
  prompt = "Read the document image carefully and return only the OCR text. Preserve original language and line breaks."
  stream = $false
  images = @($b64)
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Uri "http://127.0.0.1:11434/api/generate" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## 8. 当前项目推荐的代码用法

如果你是在本项目里新增功能，不建议重复手写 HTTP 调用。

优先复用：

- `detect_local_glm_status(profile)`
- `ocr_image_local(profile, image_path)`
- `run_e4_glm_local(profile, pages)`

原因很直接：

- 已经处理了本机地址和模型名
- 已经处理了大图降采样
- 已经处理了 JSON 解析
- 已经处理了整页 fallback

## 9. 常见问题

### 9.1 `available=False`

通常是两类问题：

1. Ollama 没启动
2. `glm-ocr:latest` 没拉取

先检查：

```powershell
ollama list
```

### 9.2 返回 `HTTP 500`

本机当前最常见原因是图片过大，尤其是很高的长条图。

处理方式：

- 先降采样
- 最长边控制到 `1024` 左右

### 9.3 为什么不用 `chat.completions`

因为这里不是文本问答模型调用，而是本地 Ollama 上的视觉 OCR 模型调用。

本项目当前走的是：

- `POST /api/generate`
- `images: [base64]`

不是 OpenAI 风格接口。

## 10. 当前项目内的验证结果

当前机器上已经确认：

- `ollama` 可用
- `glm-ocr:latest` 已拉取
- 本地最小 OCR 请求可返回文本
- 项目内 `ocr_image_local()` 可正常调用
- `run_e4_glm_local()` 在普通 profile 下可生成 E4 结果

如果后续要继续扩展，建议从 [salt_e4_glm_local.py](D:/Codes/generic/legal_understanding/salt_e4_glm_local.py) 开始，而不是重新写一套独立调用代码。
