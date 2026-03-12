"""数据集校验脚本。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline.constants import REQUIRED_JSON_FIELDS


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="校验项目中的 JSON / JSONL 数据文件。")
    parser.add_argument("path", type=str, help="待校验的数据文件路径。")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    """读取 JSON 或 JSONL 文件。"""
    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                content = line.strip()
                if not content:
                    continue
                try:
                    records.append(json.loads(content))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"第 {line_number} 行 JSON 解析失败: {exc}") from exc
        return records

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    return [data]


def detect_dataset_type(sample: dict[str, Any]) -> str:
    """根据字段推断数据类型。"""
    if {"prompt_messages", "chosen", "rejected"}.issubset(sample.keys()):
        return "dpo"
    if "messages" in sample:
        return "sft"
    if {"user_input", "expected_keypoints"}.issubset(sample.keys()):
        return "benchmark"
    if {"buggy_code", "fixed_code", "json_summary"}.issubset(sample.keys()):
        return "raw"
    return "unknown"


def validate_sft(records: list[dict[str, Any]]) -> list[str]:
    """校验 SFT 样本。"""
    issues: list[str] = []
    for index, record in enumerate(records, start=1):
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            issues.append(f"第 {index} 条 messages 数量不足。")
            continue
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if roles[:2] != ["system", "user"] or roles[-1] != "assistant":
            issues.append(f"第 {index} 条 messages 角色顺序异常: {roles}")
    return issues


def validate_dpo(records: list[dict[str, Any]]) -> list[str]:
    """校验 DPO 样本。"""
    issues: list[str] = []
    for index, record in enumerate(records, start=1):
        if not record.get("chosen") or not record.get("rejected"):
            issues.append(f"第 {index} 条 chosen/rejected 为空。")
        prompt_messages = record.get("prompt_messages")
        if not isinstance(prompt_messages, list) or len(prompt_messages) < 2:
            issues.append(f"第 {index} 条 prompt_messages 不完整。")
    return issues


def validate_raw(records: list[dict[str, Any]]) -> list[str]:
    """校验原始造数样本。"""
    issues: list[str] = []
    for index, record in enumerate(records, start=1):
        summary = record.get("json_summary", {})
        missing_fields = [field for field in REQUIRED_JSON_FIELDS if field not in summary]
        if missing_fields:
            issues.append(f"第 {index} 条缺少 JSON 摘要字段: {missing_fields}")
        variants = record.get("answer_variants", {})
        if not {"standard", "good", "bad"}.issubset(variants.keys()):
            issues.append(f"第 {index} 条 answer_variants 不完整。")
    return issues


def validate_benchmark(records: list[dict[str, Any]]) -> list[str]:
    """校验 benchmark 样本。"""
    issues: list[str] = []
    for index, record in enumerate(records, start=1):
        if not record.get("user_input"):
            issues.append(f"第 {index} 条缺少 user_input。")
        if not isinstance(record.get("expected_keypoints"), list):
            issues.append(f"第 {index} 条 expected_keypoints 不是列表。")
        if not isinstance(record.get("expected_json_fields"), list):
            issues.append(f"第 {index} 条 expected_json_fields 不是列表。")
    return issues


def main() -> None:
    """执行数据校验。"""
    args = parse_args()
    path = Path(args.path)
    records = load_records(path)
    if not records:
        raise SystemExit("文件为空，无法校验。")

    dataset_type = detect_dataset_type(records[0])
    if dataset_type == "sft":
        issues = validate_sft(records)
    elif dataset_type == "dpo":
        issues = validate_dpo(records)
    elif dataset_type == "raw":
        issues = validate_raw(records)
    elif dataset_type == "benchmark":
        issues = validate_benchmark(records)
    else:
        raise SystemExit("无法识别数据类型，请检查字段结构。")

    print(f"数据类型: {dataset_type}")
    print(f"记录数: {len(records)}")
    if "meta" in records[0] and isinstance(records[0]["meta"], dict):
        bug_counter = Counter(record["meta"].get("bug_type", "unknown") for record in records)
        print("bug_type 分布：")
        for bug_type, count in sorted(bug_counter.items()):
            print(f"  - {bug_type}: {count}")
    elif "bug_type" in records[0]:
        bug_counter = Counter(record.get("bug_type", "unknown") for record in records)
        print("bug_type 分布：")
        for bug_type, count in sorted(bug_counter.items()):
            print(f"  - {bug_type}: {count}")

    if issues:
        print("发现问题：")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)

    print("校验通过。")


if __name__ == "__main__":
    main()
