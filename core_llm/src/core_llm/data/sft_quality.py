from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def classify_instruction(instruction: str) -> str:
    text = instruction.strip()
    if "違い" in text or "比較" in text:
        return "comparison"
    procedure_markers = (
        "手順",
        "方法",
        "コツ",
        "注意点",
        "どう",
        "基本策",
        "目安",
        "役立ちますか",
        "使いますか",
    )
    if any(marker in text for marker in procedure_markers):
        return "procedure"
    return "definition"


def lint_sft_seed(path: str | Path) -> tuple[list[str], Counter[str]]:
    issues: list[str] = []
    category_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_instructions: set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id", "")).strip()
            instruction = str(row.get("instruction", "")).strip()
            output = str(row.get("output", "")).strip()

            if not row_id:
                issues.append(f"line {lineno}: missing id")
            elif row_id in seen_ids:
                issues.append(f"line {lineno}: duplicate id {row_id}")
            else:
                seen_ids.add(row_id)

            if not instruction:
                issues.append(f"line {lineno}: empty instruction")
            elif instruction in seen_instructions:
                issues.append(f"line {lineno}: duplicate instruction {instruction}")
            else:
                seen_instructions.add(instruction)

            if not output:
                issues.append(f"line {lineno}: empty output")
                continue
            if "\n" in output:
                issues.append(f"line {lineno}: multiline output")
            if len(output) < 8:
                issues.append(f"line {lineno}: output too short")
            if len(output) > 80:
                issues.append(f"line {lineno}: output too long")
            if not output.endswith("。"):
                issues.append(f"line {lineno}: output should end with 。")

            category_counts[classify_instruction(instruction)] += 1

    if category_counts["definition"] < 150:
        issues.append("too few definition examples")
    if category_counts["comparison"] < 20:
        issues.append("too few comparison examples")
    if category_counts["procedure"] < 15:
        issues.append("too few procedure examples")
    return issues, category_counts
