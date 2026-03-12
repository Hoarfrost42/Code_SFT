"""Qwen3 DPO 训练脚本。"""

from __future__ import annotations

from dataclasses import dataclass, field

from datasets import DatasetDict, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from trl import DPOConfig, DPOTrainer

from pipeline.training_utils import (
    apply_config_overrides,
    build_quantization_config,
    ensure_tokenizer_padding,
    load_json_config,
    resolve_torch_dtype,
)


@dataclass
class ScriptArguments:
    """DPO 训练参数。"""

    config_path: str | None = field(default=None, metadata={"help": "JSON 配置文件路径。"})
    model_name_or_path: str = field(
        default="./output/qwen3_sft_lora",
        metadata={"help": "SFT 阶段产出的模型或基础模型路径。"},
    )
    data_path: str = field(
        default="data/generated/processed/dpo_pairs_v2.jsonl",
        metadata={"help": "DPO 数据集路径。"},
    )
    output_dir: str = field(default="./output/qwen3_dpo_lora", metadata={"help": "模型输出目录。"})
    validation_split_ratio: float = field(default=0.05, metadata={"help": "自动切分验证集比例。"})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "每卡 batch size。"})
    per_device_eval_batch_size: int = field(default=1, metadata={"help": "每卡 eval batch size。"})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "梯度累计步数。"})
    learning_rate: float = field(default=1e-5, metadata={"help": "学习率。"})
    num_train_epochs: float = field(default=1.0, metadata={"help": "训练轮数。"})
    logging_steps: int = field(default=10, metadata={"help": "日志步数。"})
    save_steps: int = field(default=100, metadata={"help": "保存步数。"})
    eval_steps: int = field(default=100, metadata={"help": "评估步数。"})
    max_length: int = field(default=1536, metadata={"help": "总长度上限。"})
    max_prompt_length: int = field(default=1024, metadata={"help": "prompt 长度上限。"})
    dataset_num_proc: int = field(default=1, metadata={"help": "数据预处理并行数。"})
    beta: float = field(default=0.1, metadata={"help": "DPO beta。"})
    seed: int = field(default=42, metadata={"help": "随机种子。"})
    torch_dtype: str = field(default="bf16", metadata={"help": "torch dtype，可选 bf16/fp16/fp32。"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank。"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha。"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout。"})
    use_4bit: bool = field(default=False, metadata={"help": "是否开启 QLoRA 4bit。"})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "4bit 量化类型。"})
    bnb_4bit_compute_dtype: str = field(default="bf16", metadata={"help": "4bit 计算精度。"})
    report_to: str = field(default="none", metadata={"help": "日志上报目标。"})
    resume_from_checkpoint: str | None = field(default=None, metadata={"help": "断点续训 checkpoint 路径。"})
    sanity_check_only: bool = field(default=False, metadata={"help": "只做数据格式检查，不加载模型训练。"})
    trust_remote_code: bool = field(default=True, metadata={"help": "是否信任远程代码。"})


def apply_chat_prompt(example: dict[str, object], tokenizer: AutoTokenizer) -> dict[str, object]:
    """将 prompt_messages 渲染为模型 prompt。"""
    prompt = tokenizer.apply_chat_template(
        example["prompt_messages"],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompt": prompt,
        "chosen": example["chosen"],
        "rejected": example["rejected"],
        "meta": example.get("meta", {}),
    }


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

    train_dataset = train_dataset.map(
        lambda item: apply_chat_prompt(item, tokenizer),
        num_proc=script_args.dataset_num_proc,
        remove_columns=train_dataset.column_names,
        desc="Formatting DPO prompts",
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            lambda item: apply_chat_prompt(item, tokenizer),
            num_proc=script_args.dataset_num_proc,
            remove_columns=eval_dataset.column_names,
            desc="Formatting DPO prompts (eval)",
        )
    return train_dataset, eval_dataset


def main() -> None:
    """执行 DPO 训练。"""
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
    tokenizer.padding_side = "left"

    train_dataset, eval_dataset = build_dataset(script_args, tokenizer)
    print(f"训练集样本数: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"验证集样本数: {len(eval_dataset)}")
    if len(train_dataset) > 0:
        print("首条 prompt 预览：")
        print(train_dataset[0]["prompt"][:1000])

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

    evaluation_strategy = "steps" if eval_dataset is not None else "no"
    training_args = DPOConfig(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        eval_steps=script_args.eval_steps,
        eval_strategy=evaluation_strategy,
        save_strategy="steps",
        bf16=script_args.torch_dtype.lower() == "bf16",
        fp16=script_args.torch_dtype.lower() == "fp16",
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
        beta=script_args.beta,
        report_to=script_args.report_to,
        save_total_limit=3,
        remove_unused_columns=False,
        load_best_model_at_end=eval_dataset is not None,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)
    trainer.model.save_pretrained(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)
    print("DPO 训练完成。")


if __name__ == "__main__":
    main()
