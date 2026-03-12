"""基于 benchmark 对模型输出做离线评估。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pipeline.constants import REQUIRED_JSON_FIELDS


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估模型在基准集上的输出质量。")
    parser.add_argument("--benchmark-path", required=True, help="benchmark JSON/JSONL 路径。")
    parser.add_argument("--predictions-path", required=True, help="预测结果 JSONL 路径。")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    """读取 JSON 或 JSONL 文件。"""
    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                content = line.strip()
                if content:
                    records.append(json.loads(content))
        return records

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    return [data]


def extract_response(record: dict[str, Any]) -> str:
    """从预测记录中提取回答文本。"""
    for key in ("response", "prediction", "assistant_output", "output"):
        if key in record:
            return str(record[key])
    raise ValueError("预测结果缺少 response/prediction/assistant_output/output 字段。")


def extract_case_id(record: dict[str, Any]) -> str:
    """提取样本 ID。"""
    for key in ("id", "case_id", "sample_id"):
        if key in record:
            return str(record[key])
    raise ValueError("预测结果缺少 id/case_id/sample_id 字段。")


def extract_json_summary(response: str) -> dict[str, Any] | None:
    """从回答中提取最后一个 JSON 对象。"""
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", response, flags=re.DOTALL)
    candidates = fenced or re.findall(r"(\{.*\})", response, flags=re.DOTALL)
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def main() -> None:
    """执行离线评估。"""
    args = parse_args()
    benchmark_records = load_records(Path(args.benchmark_path))
    prediction_records = load_records(Path(args.predictions_path))
    predictions = {extract_case_id(record): extract_response(record) for record in prediction_records}

    total = len(benchmark_records)
    found = 0
    keypoint_hits = 0
    keypoint_total = 0
    valid_json_count = 0
    ask_more_correct = 0
    ask_more_total = 0

    for item in benchmark_records:
        case_id = str(item.get("id"))
        response = predictions.get(case_id)
        if response is None:
            continue

        found += 1
        expected_keypoints = item.get("expected_keypoints", [])
        for keypoint in expected_keypoints:
            if keypoint in response:
                keypoint_hits += 1
        keypoint_total += len(expected_keypoints)

        summary = extract_json_summary(response)
        if summary and all(field in summary for field in REQUIRED_JSON_FIELDS):
            valid_json_count += 1

        if "should_refuse_or_ask_more" in item:
            ask_more_total += 1
            expected_need_more = bool(item["should_refuse_or_ask_more"])
            predicted_need_more = bool(summary.get("need_more_context")) if summary else False
            if expected_need_more == predicted_need_more:
                ask_more_correct += 1

    coverage = found / total if total else 0.0
    keypoint_score = keypoint_hits / keypoint_total if keypoint_total else 0.0
    json_score = valid_json_count / found if found else 0.0
    ask_more_score = ask_more_correct / ask_more_total if ask_more_total else 0.0

    print(f"benchmark 总数: {total}")
    print(f"已命中预测: {found}")
    print(f"样本覆盖率: {coverage:.2%}")
    print(f"关键点命中率: {keypoint_score:.2%}")
    print(f"JSON 合法率: {json_score:.2%}")
    if ask_more_total:
        print(f"need_more_context 准确率: {ask_more_score:.2%}")


if __name__ == "__main__":
    main()
