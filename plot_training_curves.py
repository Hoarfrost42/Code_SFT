"""从 Hugging Face trainer_state.json 中绘制训练曲线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="绘制训练过程中的 loss / eval_loss / learning_rate 曲线。")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--output-dir",
        help="单个训练输出目录，例如 output/qwen3_sft_lora。",
    )
    group.add_argument(
        "--compare-dirs",
        nargs="+",
        help="多个训练输出目录，用于同图对比。",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="对比模式下每条曲线的显示名称，数量需与 compare-dirs 一致。",
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


def resolve_run_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """解析待绘制的 run 列表。"""
    if args.output_dir:
        output_dir = Path(args.output_dir)
        return [(output_dir.name, output_dir)]

    compare_dirs = [Path(path) for path in args.compare_dirs]
    if args.labels and len(args.labels) != len(compare_dirs):
        raise ValueError("--labels 的数量必须与 --compare-dirs 一致。")

    labels = args.labels or [path.name for path in compare_dirs]
    return list(zip(labels, compare_dirs))


def plot_metric(
    axis: plt.Axes,
    run_series: list[tuple[str, list[float], list[float]]],
    metric_label: str,
    palette: list[str],
) -> None:
    """在单个坐标轴上绘制某个指标。"""
    for index, (label, steps, values) in enumerate(run_series):
        if not values:
            continue
        axis.plot(
            steps,
            values,
            label=label,
            color=palette[index % len(palette)],
            linewidth=1.8,
        )
    if any(values for _, _, values in run_series):
        axis.legend()
    axis.grid(alpha=0.3)
    axis.set_ylabel(metric_label)


def main() -> None:
    """绘制训练曲线并保存。"""
    args = parse_args()
    run_specs = resolve_run_specs(args)

    if args.save_path:
        save_path = Path(args.save_path)
    elif len(run_specs) == 1:
        save_path = run_specs[0][1] / "training_curves.png"
    else:
        save_path = Path("output") / "training_curves_compare.png"

    train_series: list[tuple[str, list[float], list[float]]] = []
    eval_series: list[tuple[str, list[float], list[float]]] = []
    lr_series: list[tuple[str, list[float], list[float]]] = []

    for label, output_dir in run_specs:
        trainer_state_path = find_trainer_state(output_dir)
        log_history = load_log_history(trainer_state_path)
        train_steps, train_loss = collect_series(log_history, "loss")
        eval_steps, eval_loss = collect_series(log_history, "eval_loss")
        lr_steps, learning_rate = collect_series(log_history, "learning_rate")

        train_series.append((label, train_steps, train_loss))
        eval_series.append((label, eval_steps, eval_loss))
        lr_series.append((label, lr_steps, learning_rate))

    if not any(values for _, _, values in train_series + eval_series + lr_series):
        raise ValueError("没有在 trainer_state.json 中找到可绘制的训练日志。")

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    title = run_specs[0][0] if len(run_specs) == 1 else " / ".join(label for label, _ in run_specs)
    fig.suptitle(f"Training Curves: {title}", fontsize=14)

    palette = ["#d1495b", "#00798c", "#edae49", "#30638e", "#8f2d56", "#5c8001"]
    plot_metric(axes[0], train_series, "Train Loss", palette)
    plot_metric(axes[1], eval_series, "Eval Loss", palette)
    plot_metric(axes[2], lr_series, "Learning Rate", palette)
    axes[2].set_xlabel("Step")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    print(f"已保存训练曲线到: {save_path}")


if __name__ == "__main__":
    main()
