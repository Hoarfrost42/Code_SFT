"""在 benchmark 上运行 base / sft / dpo 模型并导出预测结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.constants import SYSTEM_PROMPT
from pipeline.training_utils import ensure_parent_dir, ensure_tokenizer_padding


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="在 benchmark 上运行推理并导出 predictions.jsonl。")
    parser.add_argument(
        "--mode",
        choices=("base", "sft", "dpo"),
        required=True,
        help="推理模式：基础模型、SFT 模型或 DPO 模型。",
    )
    parser.add_argument(
        "--benchmark-path",
        required=True,
        help="benchmark JSON 或 JSONL 路径。",
    )
    parser.add_argument(
        "--predictions-path",
        required=True,
        help="预测结果 JSONL 输出路径。",
    )
    parser.add_argument(
        "--base-model-name",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="基础模型名称。",
    )
    parser.add_argument(
        "--sft-model-path",
        default="./output/qwen3_sft_lora",
        help="SFT adapter 输出目录。",
    )
    parser.add_argument(
        "--dpo-model-path",
        default="./output/qwen3_dpo_lora",
        help="DPO adapter 输出目录。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="生成的新 token 上限。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度；默认贪心解码。",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="top-p 采样阈值。",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="bf16",
        help="模型加载精度。",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 trust_remote_code。",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    """读取 JSON 或 JSONL benchmark。"""
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


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    """解析 dtype 参数。"""
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[dtype_name]


def resolve_model_path(args: argparse.Namespace) -> str:
    """根据模式选择模型路径。"""
    if args.mode == "base":
        return args.base_model_name
    if args.mode == "sft":
        return args.sft_model_path
    return args.dpo_model_path


def build_messages(user_input: str) -> list[dict[str, str]]:
    """构造系统提示与用户输入。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]


def load_tokenizer(model_source: str, base_model_name: str, trust_remote_code: bool) -> AutoTokenizer:
    """加载 tokenizer。"""
    tokenizer_source = model_source if Path(model_source).exists() else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
    )
    ensure_tokenizer_padding(tokenizer)
    return tokenizer


def load_model(args: argparse.Namespace, model_source: str) -> Any:
    """按模式加载模型。"""
    dtype = resolve_torch_dtype(args.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "device_map": "auto",
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    if args.mode == "base":
        return AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

    return AutoPeftModelForCausalLM.from_pretrained(
        model_source,
        **model_kwargs,
    )


def generate_response(
    model: Any,
    tokenizer: AutoTokenizer,
    user_input: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """对单条样本做生成。"""
    messages = build_messages(user_input)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(prompt_text, return_tensors="pt")
    target_device = next(model.parameters()).device
    model_inputs = {key: value.to(target_device) for key, value in model_inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated = model.generate(**model_inputs, **generation_kwargs)

    prompt_length = model_inputs["input_ids"].shape[1]
    generated_tokens = generated[0][prompt_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response


def main() -> None:
    """执行 benchmark 推理。"""
    args = parse_args()
    benchmark_path = Path(args.benchmark_path)
    records = load_records(benchmark_path)
    model_source = resolve_model_path(args)
    tokenizer = load_tokenizer(model_source, args.base_model_name, args.trust_remote_code)
    model = load_model(args, model_source)
    model.eval()

    predictions: list[dict[str, Any]] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            user_input=str(record["user_input"]),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        predictions.append(
            {
                "id": str(record["id"]),
                "response": response,
                "mode": args.mode,
                "model_source": model_source,
            }
        )
        print(f"[{index}/{total}] {record['id']} done")

    ensure_parent_dir(args.predictions_path)
    with Path(args.predictions_path).open("w", encoding="utf-8") as file:
        for record in predictions:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"已导出 {len(predictions)} 条预测结果到: {args.predictions_path}")


if __name__ == "__main__":
    main()
