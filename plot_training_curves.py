"""从 Hugging Face trainer_state.json 中绘制训练曲线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="绘制训练过程中的 loss / eval_loss / learning_rate 曲线。")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="训练输出目录，例如 output/qwen3_sft_lora。",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="图片输出路径，默认保存到 output-dir/training_curves.png。",
    )
    return parser.parse_args()


def find_trainer_state(output_dir: Path) -> Path:
    """在输出目录中寻找最新的 trainer_state.json。"""
    candidates = list(output_dir.rglob("trainer_state.json"))
    if not candidates:
        raise FileNotFoundError(f"在 {output_dir} 下没有找到 trainer_state.json")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_log_history(trainer_state_path: Path) -> list[dict]:
    """读取 trainer_state.json 中的 log_history。"""
    with trainer_state_path.open("r", encoding="utf-8") as file:
        trainer_state = json.load(file)
    return trainer_state.get("log_history", [])


def collect_series(log_history: list[dict], metric_name: str) -> tuple[list[float], list[float]]:
    """提取给定指标的 step 序列。"""
    steps: list[float] = []
    values: list[float] = []
    for item in log_history:
        if metric_name not in item or "step" not in item:
            continue
        steps.append(item["step"])
        values.append(item[metric_name])
    return steps, values


def main() -> None:
    """绘制训练曲线并保存。"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    save_path = Path(args.save_path) if args.save_path else output_dir / "training_curves.png"

    trainer_state_path = find_trainer_state(output_dir)
    log_history = load_log_history(trainer_state_path)

    train_steps, train_loss = collect_series(log_history, "loss")
    eval_steps, eval_loss = collect_series(log_history, "eval_loss")
    lr_steps, learning_rate = collect_series(log_history, "learning_rate")

    if not train_loss and not eval_loss and not learning_rate:
        raise ValueError("没有在 trainer_state.json 中找到可绘制的训练日志。")

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Training Curves: {output_dir.name}", fontsize=14)

    if train_loss:
        axes[0].plot(train_steps, train_loss, label="train_loss", color="#d1495b", linewidth=1.8)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    axes[0].set_ylabel("Train Loss")

    if eval_loss:
        axes[1].plot(eval_steps, eval_loss, label="eval_loss", color="#00798c", linewidth=1.8)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    axes[1].set_ylabel("Eval Loss")

    if learning_rate:
        axes[2].plot(lr_steps, learning_rate, label="learning_rate", color="#edae49", linewidth=1.8)
        axes[2].legend()
        axes[2].grid(alpha=0.3)
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_xlabel("Step")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    print(f"已保存训练曲线到: {save_path}")


if __name__ == "__main__":
    main()
