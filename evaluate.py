"""基于 benchmark 对模型输出做离线评估。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pipeline.constants import REQUIRED_JSON_FIELDS

RuleGroup = tuple[tuple[str, ...], ...]

BUG_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "type_error": ("type_error", "typeerror", "type error", "类型错误"),
    "value_error": ("value_error", "valueerror", "value error", "值错误"),
    "index_error": ("index_error", "indexerror", "index error", "索引错误"),
    "key_error": ("key_error", "keyerror", "key error", "键错误"),
    "attribute_error": ("attribute_error", "attributeerror", "attribute error", "属性错误"),
    "module_not_found": (
        "module_not_found",
        "module not found",
        "modulenotfounderror",
        "缺少模块",
        "模块未找到",
    ),
    "import_error": ("import_error", "importerror", "import error", "导入错误"),
    "syntax_error": ("syntax_error", "syntaxerror", "syntax error", "语法错误"),
    "indentation_error": ("indentation_error", "indentationerror", "indentation error", "缩进错误"),
    "zero_division": ("zero_division", "zerodivisionerror", "division by zero", "除零错误", "被零除"),
    "logic_error": ("logic_error", "逻辑错误", "逻辑问题"),
    "insufficient_context": (
        "insufficient_context",
        "missing_context",
        "信息不足",
        "上下文不足",
        "缺少上下文",
    ),
    "mismatched_context": (
        "mismatched_context",
        "context_mismatch",
        "mismatch",
        "报错与代码不匹配",
        "上下文不匹配",
    ),
}

RELAXED_KEYPOINT_RULES: dict[str, tuple[RuleGroup, ...]] = {
    "test_001": (
        (("字符串", "str", "string"), ("整数", "int", "整型"), ("相加", "拼接", "连接", "+", "concatenate")),
        (("str(", "转成字符串", "转换为字符串", "f-string", "格式化", "f'", 'f"'),),
        (("```", "修复后的代码", "最小修复", "示例代码"),),
        (("低风险", "风险较低", "风险低"),),
    ),
    "test_002": (
        (("get_items(false)", "返回 none", "nonetype", "none"),),
        (("默认返回", "return []", "判空", "先检查", "为空判断", "兜底"),),
        (("最小修复", "小改动", "尽量小改", "不要重写"),),
    ),
    "test_003": (
        (("12.5",), ("不是合法整数", "不能直接转成 int", "invalid literal", "整数文本")),
        (("float(", "先转 float", "转为 float", "用 float"),),
        (("精度", "小数", "截断", "丢失"),),
    ),
    "test_004": (
        (("索引3", "nums[3]", "下标 3", "下标3"), ("越界", "out of range")),
        (("0到2", "0-2", "0~2", "0,1,2", "最大索引是2", "有效索引范围"),),
        (("修正索引", "检查长度", "len(", "先判断长度"),),
    ),
    "test_005": (
        (("age",), ("没有", "缺少", "不存在"), ("键", "key")),
        (("get(", "dict.get", "使用 get", "补全键", "添加 age 键", "提供默认值"),),
        (("最小修复", "小改动", "不要重写"),),
    ),
    "test_006": (
        (("list",), ("items",), ("没有", "不是", "无")), 
        (("items 是 dict 的方法", "字典的 items", "只有字典", "dict 的方法"),),
        (("enumerate", "改成字典", "数据结构改为 dict", "使用下标"),),
    ),
    "test_007": (
        (("缺少 pandas", "未安装 pandas", "没有 pandas", "no module named 'pandas'"),),
        (("pip install pandas", "安装 pandas", "安装依赖"),),
        (("只需安装", "无需重写", "不必修改业务逻辑", "不需要改代码逻辑"),),
    ),
    "test_008": (
        (("datatime",), ("拼写错误", "写错", "typo")), 
        (("datetime",),),
        (("保持原结构", "最小修改", "不改整体结构"),),
    ),
    "test_009": (
        (("除数", "b = 0", "b 为 0", "被 0"),),
        (("先检查", "除法前做检查", "避免除以 0", "if b != 0", "判零"),),
        (("保守处理", "返回默认值", "提示错误", "兜底"),),
    ),
    "test_010": (
        (("for",), ("缺少冒号", "少了冒号", "expected ':'", "结尾缺少 :")), 
        (("```", "修复后的代码", "for i in range(5):"),),
    ),
    "test_011": (
        (("缩进", "indented block", "没有正确缩进"),),
        (("补缩进", "加缩进", "只需缩进", "函数体缩进"),),
    ),
    "test_012": (
        (("缺少参数", "missing 1 required positional argument", "缺少 b"), ("b",)),
        (("补参数", "传入 b", "默认值", "给 b 设默认值"),),
        (("不要大改", "最小修复", "不改函数逻辑"),),
    ),
    "test_013": (
        (("range(1, 5)", "range(1,5)"), ("不包含 5", "不含 5", "上界不包含", "右边界不包含")),
        (("range(1, 6)", "range(1,6)"),),
        (("off-by-one", "边界错误", "少算了 5", "范围上界"),),
    ),
    "test_014": (
        (("total = 0",), ("循环内", "每次循环", "重复重置")), 
        (("移到循环外", "放到循环外", "放在 for 外"),),
        (("最小修复", "小改动"),),
    ),
    "test_015": (
        (("判断条件", "条件"), ("相反", "写反", "反了", "颠倒")),
        (("修正条件", "交换输出", "调换分支", "改成 age >= 18", "改 if/else"),),
    ),
    "test_016": (
        (("return",), ("循环外", "缩进位置不对", "首轮就返回", "过早返回")),
        (("移到循环外", "放到 for 外", "return 放到循环结束后"),),
    ),
    "test_017": (
        (("可变默认参数", "默认参数", "items=[]"), ("复用同一列表", "共享同一列表", "同一个列表", "会累积")),
        (("none",), ("初始化", "if items is none", "再创建列表", "默认值改为 none")),
        (("第二次调用", "多次调用", "上一次结果", "受第一次影响"),),
    ),
    "test_018": (
        (("浅拷贝", "copy 只是浅拷贝", "shallow copy"),),
        (("deepcopy", "深拷贝"),),
        (("共享引用", "嵌套对象仍共享", "内部列表仍共享", "引用同一个子列表"),),
    ),
    "test_019": (
        (("信息不足", "缺少信息", "上下文不足"),),
        (("报错信息", "traceback", "错误信息"), ("代码片段", "代码"), ("运行环境", "python 版本", "环境")),
        (("不要编造", "无法直接判断", "不能直接定位", "先补充信息"),),
    ),
    "test_020": (
        (("报错信息", "keyerror", "代码片段", "nums[5]"), ("不一致", "对不上", "不匹配", "矛盾")),
        (("不能直接", "无法直接", "不能唯一确定", "不能硬给", "不能直接修复"),),
        (("完整代码", "真实报错", "确认报错", "补充信息"),),
    ),
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估模型在基准集上的输出质量。")
    parser.add_argument("--benchmark-path", required=True, help="benchmark JSON/JSONL 路径。")
    parser.add_argument("--predictions-path", required=True, help="预测结果 JSONL 路径。")
    parser.add_argument(
        "--report-path",
        default="",
        help="可选的逐样本评估明细输出路径（JSONL）。",
    )
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


def normalize_text(text: str) -> tuple[str, str]:
    """生成普通文本与紧凑文本两种归一化形式。"""
    lowered = text.lower()
    translation_table = str.maketrans(
        {
            "：": ":",
            "，": ",",
            "。": ".",
            "（": "(",
            "）": ")",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "；": ";",
            "！": "!",
            "？": "?",
            "\r": " ",
            "\n": " ",
            "\t": " ",
        }
    )
    normalized = lowered.translate(translation_table)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    compact = re.sub(r"\s+", "", normalized)
    return normalized, compact


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


def contains_alias(normalized_text: str, compact_text: str, alias: str) -> bool:
    """判断回答中是否命中某个别名。"""
    alias_normalized, alias_compact = normalize_text(alias)
    return alias_normalized in normalized_text or alias_compact in compact_text


def match_rule(normalized_text: str, compact_text: str, rule: RuleGroup) -> bool:
    """判断回答是否满足一条宽松关键点规则。"""
    return all(
        any(contains_alias(normalized_text, compact_text, alias) for alias in alias_group)
        for alias_group in rule
    )


def compute_strict_keypoint_hits(response: str, expected_keypoints: list[str]) -> int:
    """按整句包含方式计算严格关键点命中数。"""
    return sum(1 for keypoint in expected_keypoints if keypoint in response)


def compute_relaxed_keypoint_hits(case_id: str, response: str, expected_keypoints: list[str]) -> tuple[int, int]:
    """按概念规则计算宽松关键点命中数。"""
    normalized_text, compact_text = normalize_text(response)
    rules = RELAXED_KEYPOINT_RULES.get(case_id)
    if not rules:
        return compute_strict_keypoint_hits(response, expected_keypoints), len(expected_keypoints)

    hits = sum(1 for rule in rules if match_rule(normalized_text, compact_text, rule))
    return hits, len(rules)


def coerce_to_bool(value: Any) -> bool:
    """将多种格式的布尔值转为 bool。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"1", "true", "yes", "y", "需要", "是"}
    return bool(value)


def match_error_type(summary: dict[str, Any] | None, expected_bug_type: str) -> bool:
    """判断 JSON 摘要中的 error_type 是否与 benchmark 标签一致。"""
    if not summary:
        return False

    predicted_error_type = str(summary.get("error_type", "")).strip()
    if not predicted_error_type:
        return False

    normalized_text, compact_text = normalize_text(predicted_error_type)
    aliases = BUG_TYPE_ALIASES.get(expected_bug_type, (expected_bug_type,))
    return any(contains_alias(normalized_text, compact_text, alias) for alias in aliases)


def write_report(report_path: Path, rows: list[dict[str, Any]]) -> None:
    """输出逐样本评估明细。"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """执行离线评估。"""
    args = parse_args()
    benchmark_records = load_records(Path(args.benchmark_path))
    prediction_records = load_records(Path(args.predictions_path))
    predictions = {extract_case_id(record): extract_response(record) for record in prediction_records}

    total = len(benchmark_records)
    found = 0
    strict_keypoint_hits = 0
    strict_keypoint_total = 0
    relaxed_keypoint_hits = 0
    relaxed_keypoint_total = 0
    valid_json_count = 0
    ask_more_correct = 0
    ask_more_total = 0
    error_type_correct = 0
    error_type_total = 0
    report_rows: list[dict[str, Any]] = []

    for item in benchmark_records:
        case_id = str(item.get("id"))
        response = predictions.get(case_id)
        if response is None:
            continue

        found += 1
        expected_keypoints = list(item.get("expected_keypoints", []))
        strict_keypoint_hits += compute_strict_keypoint_hits(response, expected_keypoints)
        strict_keypoint_total += len(expected_keypoints)

        relaxed_hits, relaxed_total = compute_relaxed_keypoint_hits(case_id, response, expected_keypoints)
        relaxed_keypoint_hits += relaxed_hits
        relaxed_keypoint_total += relaxed_total

        summary = extract_json_summary(response)
        json_valid = bool(summary and all(field in summary for field in REQUIRED_JSON_FIELDS))
        if json_valid:
            valid_json_count += 1

        expected_bug_type = str(item.get("bug_type", ""))
        if expected_bug_type:
            error_type_total += 1
            if match_error_type(summary, expected_bug_type):
                error_type_correct += 1

        ask_more_match: bool | None = None
        if "should_refuse_or_ask_more" in item:
            ask_more_total += 1
            expected_need_more = bool(item["should_refuse_or_ask_more"])
            predicted_need_more = coerce_to_bool(summary.get("need_more_context")) if summary else False
            ask_more_match = expected_need_more == predicted_need_more
            if ask_more_match:
                ask_more_correct += 1

        report_rows.append(
            {
                "id": case_id,
                "bug_type": expected_bug_type,
                "strict_keypoint_hits": compute_strict_keypoint_hits(response, expected_keypoints),
                "strict_keypoint_total": len(expected_keypoints),
                "relaxed_keypoint_hits": relaxed_hits,
                "relaxed_keypoint_total": relaxed_total,
                "json_valid": json_valid,
                "error_type_match": match_error_type(summary, expected_bug_type) if expected_bug_type else None,
                "need_more_context_match": ask_more_match,
            }
        )

    coverage = found / total if total else 0.0
    strict_keypoint_score = strict_keypoint_hits / strict_keypoint_total if strict_keypoint_total else 0.0
    relaxed_keypoint_score = relaxed_keypoint_hits / relaxed_keypoint_total if relaxed_keypoint_total else 0.0
    json_score = valid_json_count / found if found else 0.0
    ask_more_score = ask_more_correct / ask_more_total if ask_more_total else 0.0
    error_type_score = error_type_correct / error_type_total if error_type_total else 0.0

    print(f"benchmark 总数: {total}")
    print(f"已命中预测: {found}")
    print(f"样本覆盖率: {coverage:.2%}")
    print(f"严格关键点命中率: {strict_keypoint_score:.2%}")
    print(f"宽松关键点命中率: {relaxed_keypoint_score:.2%}")
    print(f"error_type 准确率: {error_type_score:.2%}")
    print(f"JSON 合法率: {json_score:.2%}")
    if ask_more_total:
        print(f"need_more_context 准确率: {ask_more_score:.2%}")

    if args.report_path:
        write_report(Path(args.report_path), report_rows)
        print(f"逐样本评估明细已导出到: {args.report_path}")


if __name__ == "__main__":
    main()
