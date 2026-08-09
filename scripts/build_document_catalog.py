"""为 docs 顶层兼容文档和 reports/ 生成稳定、可浏览的人类文档目录。

顶层报告被脚本、测试和冻结机器证据按路径引用，因此本脚本只生成按用途和
Phase 分类的 Markdown 索引，不改动原文。运行：
``python scripts/build_document_catalog.py``。
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path


TASK_RE = re.compile(r"\b(T-RISK-\d{8}-\d{2}|T\d+(?:\.\d+){1,2})\b", re.I)
LEGACY_RE = re.compile(r"^0[0-8]_.*\.md$")
CURRENT_FILES = {"README.md", "experiment_plan.md", "new_risks.md", "new_task_board.md"}

CATEGORY_META = {
    "current": ("当前入口", "当前任务、风险、实验计划和文档导航。"),
    "planning": ("规划来源", "冻结的原始规划或设计来源，不作为实时状态板。"),
    "phase0": ("Phase 0", "研究范围、文献矩阵与问题定义。"),
    "phase1": ("Phase 1", "主张、术语、参数和系统边界合同。"),
    "phase2": ("Phase 2", "物理模型、协议、仿真和硬件边界。"),
    "phase3": ("Phase 3", "解码 baseline、memory 与 oracle 对照。"),
    "phase4": ("Phase 4", "混合慢/快回路、teacher/student 与故障恢复。"),
    "phase5": ("Phase 5", "统一验证、因果消融、logical channel 与硬件 Pareto。"),
    "phase6": ("Phase 6", "Route-A、外部复现、RTL 和多证据 lane。"),
    "phase7": ("Phase 7", "论文图表/章节合同与 reviewer response。"),
    "phase9": ("Phase 9", "高保真双后端、raw-IQ 和三 lane 资格协议。"),
    "risk": ("风险任务", "插入风险任务直接产生的顶层报告。"),
    "legacy": ("旧治理链", "00—08 旧状态链，仅用于历史追溯和兼容引用。"),
    "other": ("其他文档", "尚未绑定明确 task/phase 的稳定文档。"),
}

COLLECTIONS = (
    ("new_tasks", "当前逐 task 完成记录"),
    ("reports", "按 Phase 保存的新式独立报告"),
    ("review", "历史任务与里程碑 Review"),
    ("protocols", "可执行协议和 benchmark 合同"),
    ("paper_notes", "论文草稿与装配入口"),
    ("paper_materials", "论文证据、表格和素材"),
    ("paper_readers", "离线论文阅读副本与翻译笔记"),
    ("figures", "图件生成脚本与渲染产物"),
    ("figure_assets", "图件源数据与可编辑资产"),
    ("for_human", "通俗解释、答辩和审阅说明"),
    ("deep_research_reports", "深度调研报告"),
    ("codebase_overview", "代码库结构说明"),
    ("reference", "外部调研和工具参考"),
    ("worker_summary", "旧任务交接摘要"),
    ("任务版改进记录", "用于强化 task 设计的论文笔记"),
    ("汇报用", "汇报材料"),
    ("legacy_context", "明确退役或迁移的历史材料"),
    ("tasks", "旧任务记录与兼容镜像"),
)


def relative_link(from_file: Path, to_file: Path) -> str:
    return Path(os.path.relpath(to_file, from_file.parent)).as_posix()


def title_for(path: Path, text: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return path.stem.replace("_", " ")


def primary_task(text: str, title: str) -> str | None:
    match = TASK_RE.search(title) or TASK_RE.search("\n".join(text.splitlines()[:80]))
    return match.group(1).upper() if match else None


def category_for(path: Path, task: str | None) -> str:
    if path.name in CURRENT_FILES:
        return "current"
    if LEGACY_RE.match(path.name):
        return "legacy"
    if path.name == "rough_plan.md":
        return "planning"
    if task and task.startswith("T-RISK-"):
        return "risk"
    if task:
        phase = int(task[1:].split(".", 1)[0])
        key = f"phase{phase}"
        return key if key in CATEGORY_META else "other"
    return "other"


def state_for(category: str) -> str:
    if category == "current":
        return "CURRENT"
    if category == "legacy":
        return "LEGACY"
    if category in {"planning", "other"}:
        return "REFERENCE"
    return "FROZEN"


def document_type(path: Path) -> str:
    name = path.stem.lower()
    if path.name == "README.md":
        return "导航"
    if "task_board" in name:
        return "任务板"
    if "risk" in name:
        return "风险登记"
    if "plan" in name:
        return "计划"
    if any(token in name for token in ("reviewer_response", "reviewer_explanation")):
        return "审稿回答"
    if "preregistration" in name:
        return "预注册"
    if any(token in name for token in ("contract", "ontology", "registry", "matrix")):
        return "合同/注册表"
    if any(token in name for token in ("qualification", "audit", "gate", "falsification")):
        return "审计/资格门"
    if any(token in name for token in ("validation", "reproduction", "replay")):
        return "验证/复现"
    if "baseline" in name:
        return "Baseline"
    if any(token in name for token in ("model", "decoder", "teacher", "student", "rtl")):
        return "模型/实现"
    return "报告"


def short_description(text: str, title: str) -> str:
    for line in text.splitlines()[:80]:
        candidate = " ".join(line.strip().split())
        if candidate.lstrip("#").strip() == title:
            continue
        if len(candidate) < 20:
            continue
        if candidate.startswith(("#", "|", "```", "<!--", "- [", "* [")):
            continue
        candidate = re.sub(r"^[>*-]\s*", "", candidate)
        candidate = re.sub(r"!?\[([^]]*)\]\([^)]+\)", r"\1", candidate)
        candidate = re.sub(r"\[([^]]+)\]\[[^]]+\]", r"\1", candidate)
        candidate = candidate.replace("|", "\\|")
        return candidate[:120] + ("…" if len(candidate) > 120 else "")
    return title.replace("|", "\\|")


def task_sort_key(task: str | None, filename: str) -> tuple[int, ...] | tuple[int, str]:
    if task and task.startswith("T-RISK-"):
        numbers = tuple(int(part) for part in task.removeprefix("T-RISK-").split("-"))
        return (90, *numbers)
    if task:
        return (0, *(int(part) for part in task[1:].split(".")))
    return (99, filename)


def related_files(directory: Path, task: str) -> list[Path]:
    prefix = task.upper() + "_"
    return sorted(
        path
        for path in directory.rglob("*.md")
        if path.name != "README.md"
        and (path.stem.upper() == task.upper() or path.stem.upper().startswith(prefix))
    )


def evidence_page(root: Path, task: str) -> Path | None:
    catalog = root / "docs" / "evidence_catalog"
    if task.startswith("T-RISK-"):
        page = catalog / "risk" / task.removeprefix("T-RISK-") / "README.md"
    else:
        parts = task[1:].split(".")
        if len(parts) < 2:
            return None
        page = catalog / f"phase{parts[0]}" / f"milestone_{parts[0]}_{parts[1]}" / "README.md"
    return page if page.exists() else None


def build_catalog(root: Path) -> tuple[int, int]:
    docs = root / "docs"
    output = docs / "document_catalog"
    source_paths = list(docs.glob("*.md"))
    reports = docs / "reports"
    if reports.exists():
        source_paths.extend(
            path for path in reports.rglob("*.md") if path.name != "README.md"
        )
    records = []
    for path in sorted(source_paths):
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = title_for(path, text)
        task = primary_task(text, title)
        records.append(
            {
                "path": path,
                "title": title,
                "task": task,
                "category": category_for(path, task),
                "state": state_for(category_for(path, task)),
                "type": document_type(path),
                "description": short_description(text, title),
            }
        )

    grouped = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)

    category_order = [key for key in CATEGORY_META if key in grouped]
    generated = []
    for category in category_order:
        label, description = CATEGORY_META[category]
        page = output / category / "README.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        items = sorted(
            grouped[category],
            key=lambda item: task_sort_key(item["task"], item["path"].name),
        )
        lines = [
            f"# {label} · 文档",
            "",
            description,
            "",
            f"本页索引 `{len(items)}` 个文档。每个文档均有稳定路径；从标题进入正文。",
            "",
            "| Task | 状态 | 文档 | 类型 | 关联材料 | 内容提示 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in items:
            link = relative_link(page, item["path"])
            task_value = (
                item["task"]
                if item["category"] not in {"current", "planning", "legacy"}
                else None
            )
            task = f"`{task_value}`" if task_value else "—"
            title = item["title"].replace("|", "\\|")
            related = []
            if task_value:
                completions = related_files(docs / "new_tasks", task_value)
                reviews = related_files(docs / "review", task_value)
                evidence = evidence_page(root, task_value)
                if completions:
                    label = "完成记录" if len(completions) == 1 else f"完成记录×{len(completions)}"
                    related.append(f"[{label}]({relative_link(page, completions[0])})")
                if reviews:
                    label = "Review" if len(reviews) == 1 else f"Review×{len(reviews)}"
                    related.append(f"[{label}]({relative_link(page, reviews[0])})")
                if evidence:
                    related.append(f"[机器证据]({relative_link(page, evidence)})")
            lines.append(
                f"| {task} | `{item['state']}` | [{title}]({link}) | {item['type']} | {' · '.join(related) or '—'} | {item['description']} |"
            )
        page.write_text("\n".join(lines) + "\n", encoding="utf-8")
        generated.append(page)

    index = output / "README.md"
    lines = [
        "# 人类文档目录",
        "",
        "这里是 `docs/` 顶层兼容文档与 `reports/` 的人类可读视图。高绑定原文继续保留旧路径，以免破坏生成脚本、测试、冻结 JSON/CSV、manifest 和历史引用；本目录按 current / phase / risk / legacy 重新组织阅读入口。",
        "",
        "## 推荐阅读顺序",
        "",
        "1. 从“当前入口”确认任务板、风险和实验计划。",
        "2. 按 Phase 打开主题报告；表格中的 Task 是与任务板对齐的主任务 ID。",
        "3. 需要机器证据时转到 [`../evidence_catalog/README.md`](../evidence_catalog/README.md)。",
        "4. `legacy` 只用于追溯，不可覆盖当前状态源。",
        "",
        "## 分类",
        "",
        "| 分类 | 文档数 | 用途 |",
        "| --- | ---: | --- |",
    ]
    for category in category_order:
        label, description = CATEGORY_META[category]
        page = output / category / "README.md"
        lines.append(
            f"| [{label}]({relative_link(index, page)}) | {len(grouped[category])} | {description} |"
        )
    lines.extend(
        [
            "",
            "## 专题目录",
            "",
            "| 目录 | Markdown 数 | 角色 |",
            "| --- | ---: | --- |",
        ]
    )
    for dirname, role in COLLECTIONS:
        directory = docs / dirname
        readme = directory / "README.md"
        if not readme.exists():
            continue
        count = sum(1 for _ in directory.rglob("*.md"))
        lines.append(
            f"| [{dirname}/]({relative_link(index, readme)}) | {count} | {role} |"
        )
    lines.extend(
        [
            "",
            "## 路径政策",
            "",
            "- 未迁移的顶层 Markdown 是当前兼容层，不再作为人工浏览入口。",
            "- 新 task 完成记录写入 `docs/new_tasks/`；新的独立人类报告写入 `docs/reports/phaseN/`。",
            "- 只有解除代码、测试、机器证据和哈希绑定后，才物理迁移旧报告。",
            "- 文档新增或状态变化后运行 `python scripts/build_document_catalog.py` 刷新本目录。",
        ]
    )
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(records), len(generated) + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    document_count, page_count = build_catalog(args.root.resolve())
    print(f"document_count={document_count}")
    print(f"generated_page_count={page_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
