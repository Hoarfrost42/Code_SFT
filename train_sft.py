"""Qwen3 SFT 训练脚本。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from pipeline.training_utils import (
    apply_config_overrides,
    build_quantization_config,
    ensure_tokenizer_padding,
    load_json_config,
    resolve_torch_dtype,
)


@dataclass
class ScriptArguments:
    """SFT 训练参数。"""

    config_path: str | None = field(default=None, metadata={"help": "JSON 配置文件路径。"})
    model_name_or_path: str = field(
        default="Qwen/Qwen3-4B-Instruct-2507",
        metadata={"help": "基础模型或本地模型路径。"},
    )
    data_path: str = field(
        default="data/final/processed/sft_full_v1.jsonl",
        metadata={"help": "SFT 数据集路径。"},
    )
    output_dir: str = field(default="./output/qwen3_sft_lora", metadata={"help": "模型输出目录。"})
    validation_split_ratio: float = field(default=0.05, metadata={"help": "自动切分验证集比例。"})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "每卡 batch size。"})
    per_device_eval_batch_size: int = field(default=1, metadata={"help": "每卡 eval batch size。"})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "梯度累计步数。"})
    learning_rate: float = field(default=5e-5, metadata={"help": "学习率。"})
    weight_decay: float = field(default=0.01, metadata={"help": "权重衰减。"})
    num_train_epochs: float = field(default=3.0, metadata={"help": "训练轮数。"})
    logging_steps: int = field(default=10, metadata={"help": "日志步数。"})
    save_steps: int = field(default=100, metadata={"help": "保存步数。"})
    eval_steps: int = field(default=100, metadata={"help": "评估步数。"})
    max_seq_length: int = field(default=1536, metadata={"help": "最大序列长度。"})
    dataset_num_proc: int = field(default=1, metadata={"help": "数据预处理并行数。"})
    seed: int = field(default=42, metadata={"help": "随机种子。"})
    torch_dtype: str = field(default="bf16", metadata={"help": "torch dtype，可选 bf16/fp16/fp32。"})
    lora_r: int = field(default=32, metadata={"help": "LoRA rank。"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha。"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout。"})
    use_4bit: bool = field(default=False, metadata={"help": "是否开启 QLoRA 4bit。"})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "4bit 量化类型。"})
    bnb_4bit_compute_dtype: str = field(default="bf16", metadata={"help": "4bit 计算精度。"})
    report_to: str = field(default="none", metadata={"help": "日志上报目标。"})
    response_template: str = field(
        default="<|im_start|>assistant\n",
        metadata={"help": "assistant 回复起始模板。"},
    )
    resume_from_checkpoint: str | None = field(default=None, metadata={"help": "断点续训 checkpoint 路径。"})
    sanity_check_only: bool = field(default=False, metadata={"help": "只做数据格式检查，不加载模型训练。"})
    trust_remote_code: bool = field(default=True, metadata={"help": "是否信任远程代码。"})


def apply_chat_template(example: dict[str, object], tokenizer: AutoTokenizer) -> dict[str, str]:
    """将 messages 转换为模型输入文本。"""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def build_dataset(script_args: ScriptArguments, tokenizer: AutoTokenizer) -> tuple[object, object | None]:
    """构建训练集和验证集。"""
    dataset = load_dataset("json", data_files=script_args.data_path, split="train")
    if script_args.validation_split_ratio > 0:
        split_dataset: DatasetDict = dataset.train_test_split(
            test_size=script_args.validation_split_ratio,
            seed=script_args.seed,
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    map_kwargs = {
        "num_proc": script_args.dataset_num_proc,
        "remove_columns": train_dataset.column_names,
        "desc": "Applying chat template",
    }
    train_dataset = train_dataset.map(lambda item: apply_chat_template(item, tokenizer), **map_kwargs)

    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            lambda item: apply_chat_template(item, tokenizer),
            num_proc=script_args.dataset_num_proc,
            remove_columns=eval_dataset.column_names,
            desc="Applying chat template (eval)",
        )

    return train_dataset, eval_dataset


def main() -> None:
    """执行 SFT 训练。"""
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    config = load_json_config(script_args.config_path)
    script_args = apply_config_overrides(script_args, config)

    set_seed(script_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
    )
    ensure_tokenizer_padding(tokenizer)

    train_dataset, eval_dataset = build_dataset(script_args, tokenizer)
    print(f"训练集样本数: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"验证集样本数: {len(eval_dataset)}")
    if len(train_dataset) > 0:
        print("首条格式化样本预览：")
        print(train_dataset[0]["text"][:1000])

    if script_args.sanity_check_only:
        print("已完成 sanity check，未加载模型。")
        return

    quantization_config = build_quantization_config(
        load_in_4bit=script_args.use_4bit,
        quant_type=script_args.bnb_4bit_quant_type,
        compute_dtype=script_args.bnb_4bit_compute_dtype,
    )

    model_kwargs = {
        "trust_remote_code": script_args.trust_remote_code,
        "device_map": "auto",
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = resolve_torch_dtype(script_args.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if script_args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    response_template_ids = tokenizer.encode(
        script_args.response_template,
        add_special_tokens=False,
    )
    collator = None
    if response_template_ids:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer,
        )
    else:
        print("警告：response_template 无法编码，将退回全量 loss。")

    eval_strategy = "steps" if eval_dataset is not None else "no"
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        weight_decay=script_args.weight_decay,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        eval_steps=script_args.eval_steps,
        eval_strategy=eval_strategy,
        save_strategy="steps",
        bf16=script_args.torch_dtype.lower() == "bf16",
        fp16=script_args.torch_dtype.lower() == "fp16",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        report_to=script_args.report_to,
        save_total_limit=3,
        remove_unused_columns=False,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False if eval_dataset is not None else None,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        packing=False,
    )

    trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)
    trainer.model.save_pretrained(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)
    print("SFT 训练完成。")


if __name__ == "__main__":
    main()
