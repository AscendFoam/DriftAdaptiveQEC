"""为 docs 顶层机器证据生成按阶段组织的人类可读目录。

旧 JSON/CSV 带有路径、自哈希和 release-pin 绑定，因此目录只生成 Markdown
索引，不改动证据文件本身。运行：``python scripts/build_evidence_catalog.py``。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PHASE_RE = re.compile(r"^t(?P<phase>\d+)_(?P<milestone>\d+)_")
TASK_RE = re.compile(r"^(t\d+_\d+_\d+)")
RISK_RE = re.compile(r"^t_risk_(?P<date>\d{8})_(?P<sequence>\d{2})_")


def category_for(path: Path) -> tuple[str, str]:
    risk = RISK_RE.match(path.name)
    if risk:
        risk_id = f"{risk.group('date')}-{risk.group('sequence')}"
        return "risk", risk_id
    phase = PHASE_RE.match(path.name)
    if phase:
        number = phase.group("phase")
        milestone = phase.group("milestone")
        return f"phase{number}", f"milestone_{number}_{milestone}"
    return "contracts", "core_contracts"


def task_key(path: Path) -> str:
    risk = RISK_RE.match(path.name)
    if risk:
        return f"T-RISK-{risk.group('date')}-{risk.group('sequence')}"
    task = TASK_RE.match(path.name)
    if task:
        return task.group(1).upper().replace("_", ".")
    return "CORE"


def format_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MiB"
    if size >= 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size} B"


def json_summary(path: Path) -> str:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "JSON 证据（当前快照中不可直接解析或为 LFS pointer）"
    if not isinstance(payload, dict):
        return "JSON 机器证据"

    parts = []
    for key in ("task_id", "protocol_id", "artifact_type", "status", "verdict"):
        value = payload.get(key)
        if isinstance(value, (str, int, float, bool)) and str(value):
            parts.append(f"{key}={value}")
    for key in ("title", "purpose", "objective"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            text = " ".join(value.split())
            parts.append(text[:90] + ("…" if len(text) > 90 else ""))
            break
    return "；".join(parts[:4]) or "JSON 机器证据"


def csv_summary(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
    except (OSError, UnicodeDecodeError, csv.Error):
        return "CSV Source Data"
    columns = ", ".join(header[:4])
    if len(header) > 4:
        columns += ", …"
    return f"{rows:,} rows；{len(header)} columns（{columns}）"


def best_human_doc(
    root: Path, artifact: Path, markdown_cache: dict[Path, str]
) -> Path | None:
    exact = artifact.relative_to(root).as_posix()
    searches = [(exact, artifact.name)]
    if task_key(artifact) != "CORE":
        searches.append((task_key(artifact),))
    for needles in searches:
        candidates = []
        for path, text in markdown_cache.items():
            if not any(needle in text or needle in path.name for needle in needles):
                continue
            relative = path.relative_to(root)
            if relative.parts[:2] in {
                ("docs", "legacy_context"),
                ("docs", "tasks"),
            }:
                continue
            location_rank = 0 if path.parent == root / "docs" else 2
            if relative.parts[:2] == ("docs", "new_tasks"):
                location_rank = 1
            score = (
                location_rank,
                1 if path.name in {"README.md", "new_task_board.md"} else 0,
                len(relative.parts),
                len(text),
            )
            candidates.append((score, path))
        if candidates:
            return min(candidates)[1]
    return None


def relative_link(from_file: Path, to_file: Path) -> str:
    return Path(os.path.relpath(to_file, from_file.parent)).as_posix()


def category_sort_key(item: tuple[str, str]) -> tuple[int, int, int, str]:
    category, section = item
    if category == "contracts":
        return (0, 0, 0, section)
    if category.startswith("phase"):
        phase = int(category.removeprefix("phase"))
        milestone = int(section.rsplit("_", 1)[-1])
        return (1, phase, milestone, section)
    date, sequence = section.split("-", 1)
    return (2, int(date), int(sequence), section)


def category_label(category: str, section: str) -> str:
    if category == "contracts":
        return "核心合同"
    if category.startswith("phase"):
        phase = category.removeprefix("phase")
        milestone = section.rsplit("_", 1)[-1]
        return f"Milestone {phase}.{milestone}"
    return f"T-RISK-{section}"


def build_catalog(root: Path) -> tuple[int, int]:
    docs = root / "docs"
    output = docs / "evidence_catalog"
    artifacts = sorted(
        path
        for path in docs.iterdir()
        if path.is_file() and path.suffix.lower() in {".json", ".csv"}
    )
    markdown_cache = {
        path: path.read_text(encoding="utf-8", errors="ignore")
        for path in docs.rglob("*.md")
        if output not in path.parents
    }

    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for artifact in artifacts:
        grouped[category_for(artifact)].append(artifact)

    generated = []
    for (category, section), paths in sorted(
        grouped.items(), key=lambda item: category_sort_key(item[0])
    ):
        page = output / category / section / "README.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        title = (
            f"Phase {category.removeprefix('phase')} · {category_label(category, section)}"
            if category.startswith("phase")
            else (f"风险任务 {section}" if category == "risk" else "核心机器合同")
        )
        lines = [
            f"# {title}",
            "",
            f"本页索引 `{len(paths)}` 个冻结机器证据。文件仍保留原路径，以维持自哈希和 release-pin；优先阅读“人类文档”列。",
            "",
            "| Task | 机器证据 | 内容概览 | 人类文档 |",
            "| --- | --- | --- | --- |",
        ]
        for path in paths:
            link = relative_link(page, path)
            size = format_size(path.stat().st_size)
            summary = json_summary(path) if path.suffix == ".json" else csv_summary(path)
            summary = summary.replace("|", "\\|").replace("\n", " ")
            human = best_human_doc(root, path, markdown_cache)
            human_link = (
                f"[{human.name}]({relative_link(page, human)})" if human else "—"
            )
            lines.append(
                f"| `{task_key(path)}` | [{path.name}]({link}) · {size} | {summary} | {human_link} |"
            )
        page.write_text("\n".join(lines) + "\n", encoding="utf-8")
        generated.append(page)

    index = output / "README.md"
    lines = [
        "# 机器证据目录",
        "",
        "这里是 `docs/` 顶层 JSON/CSV 的人类可读投影。旧证据保持原路径和原字节，避免破坏路径绑定、自哈希、release pin 与论文证据链；本目录只提供按 phase / milestone / risk 分类的导航。",
        "",
        "## 阅读顺序",
        "",
        "1. 先打开具体 phase 或 risk 页面。",
        "2. 优先阅读每行的“人类文档”。",
        "3. 需要复核数字、schema 或 provenance 时，再打开 JSON/CSV。",
        "",
        "## 分类导航",
    ]
    previous_category = None
    for (category, section), paths in sorted(
        grouped.items(), key=lambda item: category_sort_key(item[0])
    ):
        if category != previous_category:
            heading = (
                "核心合同"
                if category == "contracts"
                else (
                    f"Phase {category.removeprefix('phase')}"
                    if category.startswith("phase")
                    else "风险任务"
                )
            )
            lines.extend(
                [
                    "",
                    f"### {heading}",
                    "",
                    "| 分组 | 文件数 | 索引页 |",
                    "| --- | ---: | --- |",
                ]
            )
            previous_category = category
        page = output / category / section / "README.md"
        lines.append(
            f"| {category_label(category, section)} | {len(paths)} | [打开]({relative_link(index, page)}) |"
        )
    lines.extend(
        [
            "",
            "## 路径政策",
            "",
            "- 现有顶层 JSON/CSV 是冻结证据：只在对应任务重新生成并重新封存时迁移。",
            "- 新任务不得继续向 `docs/` 顶层增加机器文件，应写入 `docs/evidence/<phase>/<milestone>/`。",
            "- VS Code 文件树默认隐藏 `docs/*.json` 和 `docs/*.csv`；需要查看时可临时关闭 `files.exclude`。",
            "- 更新证据后运行 `python scripts/build_evidence_catalog.py` 刷新本目录。",
        ]
    )
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(artifacts), len(generated) + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    artifact_count, page_count = build_catalog(args.root.resolve())
    print(f"artifact_count={artifact_count}")
    print(f"generated_page_count={page_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
