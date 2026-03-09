import os
import torch
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-3B-Instruct", 
        metadata={"help": "The model that you want to train from the Hugging Face hub or local path."}
    )
    data_path: str = field(
        default="data/SFT_data/exports/sft_train_v1.jsonl",
        metadata={"help": "Path to the training data jsonl file."}
    )
    output_dir: str = field(
        default="./output/sft_qwen",
        metadata={"help": "Where to store the final model."}
    )
    per_device_train_batch_size: int = field(default=2, metadata={"help": "Batch size per GPU."})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    learning_rate: float = field(default=2e-4, metadata={"help": "Initial learning rate (AdamW)."})
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    logging_steps: int = field(default=5, metadata={"help": "Log every X updates steps."})
    max_seq_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})
    lora_r: int = field(default=16, metadata={"help": "LoRA R value."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})

def apply_chat_template_to_dataset(example, tokenizer):
    """
    将 JSONL 中的 messages array 转换为模型能够理解的格式化的对话 prompt。
    针对基于 chat template 的 Qwen 模型。
    """
    messages = example["messages"]
    # 使用 tokenizer 预设的 template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": prompt}

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    set_seed(42)

    print(f"Loading data from {script_args.data_path}...")
    dataset = load_dataset("json", data_files=script_args.data_path, split="train")

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, 
        trust_remote_code=True
    )
    # 确保 pad token 不为空
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
    
    print("Formatting dataset mapping...")
    dataset = dataset.map(
        lambda x: apply_chat_template_to_dataset(x, tokenizer),
        num_proc=4,
        remove_columns=dataset.column_names,
        desc="Applying chat template"
    )

    print(f"Loading model {script_args.model_name_or_path} ...")
    # 为了 24G 4090，我们采取 bf16 混合精度加载，暂且不开启 4-bit（4B 模型参数约等于8GB，加上梯度在 24G 卡上是跑得动的）
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    # Enable gradient checkpointing for memory saving
    model.gradient_checkpointing_enable()
    # If 4-bit/8-bit needed in future, uncomment prepare_model_for_kbit_training
    # model = prepare_model_for_kbit_training(model)

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 训练配置
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_strategy="epoch",
        bf16=True,                       # 4090 支持 bfloat16，速度更快且防溢出
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        report_to="none"                 # 取消wandb等追踪上报
    )

    # Note: Qwen 等模型多以 "<|im_start|>assistant\n" 来标识回答开始，DataCollatorForCompletionOnlyLM可屏蔽 user prompt 算 loss
    # 这里我们采用常规形式，或者指定 response template
    response_template = "<|im_start|>assistant\n"
    if response_template not in tokenizer.vocab:
        # 兼容备选模式
        collator = None 
        print("Warning: default Completion collator ignored due to tokenization specifics.")
    else:
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.model.save_pretrained(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)
    print("All Done!")

if __name__ == "__main__":
    main()
