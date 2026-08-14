"""One-off, auditable migration from flat experiment files to the approved layout."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "codefiles"
DOCS = ROOT / "docs"
LEGACY = CODE / "legacy"
LOG_PATH = ROOT / "results" / "repository_migration" / "layout_migration.json"


def move(source: Path, destination: Path, operations: list[dict[str, str]]) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    source.replace(destination)
    operations.append(
        {
            "action": "move",
            "source": str(source.relative_to(ROOT)),
            "destination": str(destination.relative_to(ROOT)),
        }
    )


def main() -> None:
    operations: list[dict[str, str]] = []
    LEGACY.mkdir(parents=True, exist_ok=True)
    for script in sorted(CODE.glob("*.py")):
        if not re.fullmatch(r"\d{2}_.+\.py", script.name):
            continue
        destination = LEGACY / script.name
        move(script, destination, operations)
        text = destination.read_text(encoding="utf-8")
        updated = text.replace("Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parents[2]")
        destination.write_text(updated, encoding="utf-8")
        operations.append({"action": "rewrite_project_root", "path": str(destination.relative_to(ROOT))})

    moves = [
        (
            DOCS / "煤气发电量预测与发电优化-2.pdf",
            DOCS / "competition" / "official" / "煤气发电量预测与发电优化-2.pdf",
        ),
        (DOCS / "modeling_idea.md", DOCS / "design" / "modeling_idea.md"),
        (
            DOCS / "experimental_docs" / "01_题目阅读与数据预处理实验报告.md",
            DOCS / "experimental_docs" / "preprocessing" / "01_题目阅读与数据预处理实验报告.md",
        ),
        (
            DOCS / "experimental_docs" / "02_训练集插值与全空列分析实验报告.md",
            DOCS / "experimental_docs" / "preprocessing" / "02_训练集插值与全空列分析实验报告.md",
        ),
        (
            DOCS / "experimental_docs" / "03_多步发电预测与时间因素实验报告.md",
            DOCS / "experimental_docs" / "forecasting" / "03_多步发电预测与时间因素实验报告.md",
        ),
        (
            DOCS / "experimental_docs" / "04_数据清洗再审计与预测优化实验报告.md",
            DOCS / "experimental_docs" / "forecasting" / "04_数据清洗再审计与预测优化实验报告.md",
        ),
        (
            DOCS / "experimental_docs" / "05_发电优化约束可辨识性审计报告.md",
            DOCS / "experimental_docs" / "optimization" / "05_发电优化约束可辨识性审计报告.md",
        ),
        (
            DOCS / "experimental_docs" / "06_保守软约束发电优化与历史回放实验报告.md",
            DOCS / "experimental_docs" / "optimization" / "06_保守软约束发电优化与历史回放实验报告.md",
        ),
    ]
    for source, destination in moves:
        move(source, destination, operations)

    for report in [
        DOCS / "experimental_docs" / "forecasting" / "04_数据清洗再审计与预测优化实验报告.md",
        DOCS / "experimental_docs" / "optimization" / "06_保守软约束发电优化与历史回放实验报告.md",
    ]:
        if report.exists():
            text = report.read_text(encoding="utf-8")
            report.write_text(text.replace("../../results/", "../../../results/"), encoding="utf-8")
            operations.append({"action": "rewrite_relative_result_links", "path": str(report.relative_to(ROOT))})

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        json.dumps(
            {"created_utc": datetime.now(timezone.utc).isoformat(), "operations": operations},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Completed {len(operations)} migration operations; log={LOG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

