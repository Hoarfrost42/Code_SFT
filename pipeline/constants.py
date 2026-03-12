"""项目级常量定义。"""

from __future__ import annotations

SYSTEM_PROMPT = """你是一位专业的代码诊断与修复助手。你的任务是审查用户提供的报错信息与代码片段，并给出简洁、可靠、可执行的修复建议。

请严格按照以下顺序回答：
1. 错误类型与根因分析
2. 最小修复建议
3. 修复后的代码片段
4. 风险提示
5. 输出 JSON 摘要

要求：
- 优先采用最小改动，而不是重写整段代码
- 不要臆造不存在的上下文、变量、依赖或运行环境
- 如果信息不足或报错与代码不匹配，要明确说明，并指出需要补充哪些信息
- 保持简洁、清晰、工程化
- JSON 摘要必须包含以下字段：error_type, root_cause, fix_summary, risk_note, need_more_context, confidence"""

REQUIRED_JSON_FIELDS = (
    "error_type",
    "root_cause",
    "fix_summary",
    "risk_note",
    "need_more_context",
    "confidence",
)

CONFIDENCE_LEVELS = ("low", "medium", "high")

ERROR_STYLE_TEMPLATES = (
    "报错：{error_message}\n代码片段：\n{code}\n请定位错误并给出最小修复。",
    "运行时报错了。\n异常信息：{error_message}\n代码如下：\n{code}\n请帮我最小修复。",
    "这段 Python 代码报错了：{error_message}\n代码：\n{code}\n要求：优先最小修改。",
)

LOGIC_STYLE_TEMPLATES = (
    "程序应该输出 {expected_output}，但是输出了 {actual_output}。\n代码片段：\n{code}",
    "没有报错，但结果不对。\n期望输出：{expected_output}\n实际输出：{actual_output}\n代码：\n{code}",
    "这段代码能运行，但是结果不符合预期。\n预期：{expected_output}\n实际：{actual_output}\n代码片段：\n{code}\n请只做最小修复。",
)

MISSING_CONTEXT_PROMPTS = (
    "代码跑不动了，咋办？",
    "这段 Python 有问题，你帮我看看。",
    "程序出错了，但我现在只记得它不工作了。",
)

MISMATCH_PROMPTS = (
    "报错：{error_message}\n代码片段：\n{code}\n我感觉报错和代码对不上，你帮我判断一下。",
    "这是别人发给妈妈的代码和报错，但我怀疑不是一回事：\n报错：{error_message}\n代码：\n{code}",
)

