import json
import os
import uuid
import random

# TODO: 替换为您实际请求 LLM API (如 OpenAI/Qwen/Local) 的逻辑
def call_llm_api(prompt, temp=0.7):
    """
    假装调用一个高智商 LLM (如 GPT-4, Qwen Max) 去执行生成任务。
    由于没有具体秘钥，这里用伪代码框架占个位，您可以通过对接真实的 API 完成批量造数。
    """
    pass

# 数据生成的核心指导 Prompt
SYNTHETIC_PROMPT = """
你现在是一个 Python 代码题目出题专家。请帮我生成 10 个不同的、包含各种常见或边缘 Python 错误的“用户求助输入”。
随后，针对每个输入，严格按照如下格式同时给我生成 3 个不同质量的 Assistant 回答：
1. [Standard]：完美的标准回答。必须包含 1.错误类型与根因分析 2.最小修复建议 3.修复后的代码片段 4.风险提示 5.严格包含 6 个字段的输出 JSON 摘要。
2. [Good]：较好的回答。诊断基本正确，但缺少风险提示，且 JSON 摘要不全（如遗漏了 confidence 字段）。
3. [Error]：错误的回答。故意包含幻觉、或者把对的代码改错、或者是废话连篇，提供错误的 JSON 配置。

你需要将最后生成的结果打包成一个合规的 JSON 数组结构。
"""

def generate_bulk_data(output_dir, batch_count=5):
    """
    使用大模型合成更多样本。
    batch_count = 5 意味着请求 5 次 大模型，每次如果生成 10 个样本，那么就有 50x3 = 150 条数据。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_generated_items = []
    
    for i in range(batch_count):
        print(f"正在进行第 {i+1}/{batch_count} 批次生成...")
        
        # 实际情况中： raw_text = call_llm_api(SYNTHETIC_PROMPT, temp=0.7 + i*0.05)
        # 然后将 raw_text 里的 JSON 字符串反序列化
        
        # 由于我们无法直接在本地挂接真实的 API 并等待返回，这里提供一键式脚手架思路：
        # 1. 挂接 openai 库，请求 gpt-4o 或 qwen-max
        # 2. 从 response 取出 JSON 解析。
        # 3. 为他们分配 uuid 生成 `gold_{uuid}_standard` 的 meta_id。
        # 4. all_generated_items.extend(parsed_list)
        pass

    # 将新合成的数百条数据，利用之前写过的 split_data 逻辑进行拆分，也可以直接在这里生成三个仓。
    # write_jsonl(...)

if __name__ == "__main__":
    generate_bulk_data("./data/SFT_data/synthetic")
    print("此文件为占位生成脚本（generate_synthetic_data.py），待接入具体 API Key 后运作。")
