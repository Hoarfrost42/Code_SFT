"""训练与数据处理公共工具。"""

from __future__ import annotations

import json
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

import torch
from transformers import BitsAndBytesConfig


def load_json_config(config_path: str | None) -> dict[str, Any]:
    """读取 JSON 配置文件。"""
    if not config_path:
        return {}

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def apply_config_overrides(args_obj: Any, config: dict[str, Any]) -> Any:
    """应用配置文件，但保留命令行中显式传入的值。"""
    field_map = {field.name: field for field in fields(args_obj)}
    for key, value in config.items():
        field_info = field_map.get(key)
        if field_info is None:
            continue

        if field_info.default is not MISSING:
            default_value = field_info.default
        elif field_info.default_factory is not MISSING:
            default_value = field_info.default_factory()
        else:
            default_value = getattr(args_obj, key)

        current_value = getattr(args_obj, key)
        if current_value == default_value:
            setattr(args_obj, key, value)
    return args_obj


def ensure_parent_dir(file_path: str) -> None:
    """确保目标文件的父目录存在。"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def ensure_tokenizer_padding(tokenizer: Any) -> None:
    """为 tokenizer 设置 pad token。"""
    if tokenizer.pad_token is not None:
        return

    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token = "<|endoftext|>"


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """将字符串映射为 torch dtype。"""
    mapping = {
        "auto": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_name.lower(), torch.float32)


def build_quantization_config(
    load_in_4bit: bool,
    quant_type: str,
    compute_dtype: str,
) -> BitsAndBytesConfig | None:
    """按需构造 4-bit 量化配置。"""
    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=resolve_torch_dtype(compute_dtype),
    )


def dump_jsonl(records: list[dict[str, Any]], output_path: str) -> None:
    """以 UTF-8 编码写入 JSONL。"""
    ensure_parent_dir(output_path)
    with Path(output_path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
