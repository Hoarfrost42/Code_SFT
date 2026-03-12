# Qwen3-4B 单文件 Python 诊断微调项目

这个仓库对应一个范围明确的小任务：用户输入单文件 Python 代码片段，以及报错信息或“结果不对”的描述，模型输出结构化的排查结论和最小修复建议。本项目基于 `Qwen/Qwen3-4B-Instruct-2507` 进行 SFT 与 DPO 训练。

目标是使小模型能够回答简单的代码问题，进而迭代成一个简单插件。模型的回答需要尽量满足这几件事：

- 根因判断尽量准确
- 修复方案尽量小改，不随意重写
- 在信息不足时明确追问，不臆造上下文
- 最终输出固定的 JSON 摘要字段

## 任务边界

当前版本只处理单文件 Python 片段，典型输入包括：

- traceback + 代码
- 没有异常，但结果和预期不一致
- 代码片段与报错不匹配
- 信息不足，需要先补上下文

固定的 JSON 摘要字段如下：

- `error_type`
- `root_cause`
- `fix_summary`
- `risk_note`
- `need_more_context`
- `confidence`

## 数据说明

训练数据已经放在仓库中，可以直接使用，不需要再额外生成。

主要文件：

- [sft_full_v1.jsonl](/D:/LLM_Learning/SFT/data/final/processed/sft_full_v1.jsonl)
- [dpo_full_v1.jsonl](/D:/LLM_Learning/SFT/data/final/processed/dpo_full_v1.jsonl)
- [candidate_good_full_v1.jsonl](/D:/LLM_Learning/SFT/data/final/processed/candidate_good_full_v1.jsonl)
- [full_cases_v1.jsonl](/D:/LLM_Learning/SFT/data/final/raw/full_cases_v1.jsonl)
- [benchmark_dev_v1.json](/D:/LLM_Learning/SFT/data/benchmark/benchmark_dev_v1.json)
- [benchmark_test_v1.json](/D:/LLM_Learning/SFT/data/benchmark/benchmark_test_v1.json)

这批数据的来源可以理解为“教师模型蒸馏 + 人工筛选 + 偏好构造”：

- 一部分是人工整理过的高质量种子样本
- 一部分是围绕这些种子扩展出来的教师风格监督数据
- 一部分是为了补齐错误类型分布而加入的受控多样性样本

如果说得严格一点，这里更接近 `response distillation / supervision distillation`，而不是传统意义上蒸馏 logits 的那种知识蒸馏。不过在项目文档里写成“由强教师模型蒸馏而来，并经人工筛选清洗”是说得通的。

当前可直接训练的数据规模：

- SFT：`1824` 条
- DPO：`1824` 条

覆盖的主要错误类型包括：

- `type_error`
- `value_error`
- `index_error`
- `key_error`
- `attribute_error`
- `module_not_found`
- `import_error`
- `syntax_error`
- `indentation_error`
- `zero_division`
- `logic_error`
- `fstring_error`
- `missing_context`
- `mismatched_context`

## 目录说明

当前仓库只保留训练、评测和最终数据相关文件，结构尽量收紧：

```text
.
├─ configs/
├─ data/
│  ├─ final/
│  │  ├─ processed/
│  │  └─ raw/
│  └─ train_data/
├─ pipeline/
├─ evaluate.py
├─ plot_training_curves.py
├─ run_benchmark_inference.py
├─ train_sft.py
├─ train_dpo.py
├─ validate_dataset.py
├─ run_sft.sh
├─ run_dpo.sh
└─ README.md
```

其中：

- `train_sft.py`：LoRA SFT 训练脚本
- `train_dpo.py`：LoRA DPO 训练脚本
- `validate_dataset.py`：数据结构校验
- `evaluate.py`：离线 benchmark 评测
- `plot_training_curves.py`：训练过程曲线可视化
- `configs/`：训练配置
- `pipeline/`：训练和校验仍在使用的公共模块
- `data/train_data/data_v1.json`：benchmark 种子集
- `data/benchmark/`：显式划分好的 dev/test benchmark

## 训练流程

推荐训练顺序很直接：

1. 先做 SFT，让模型学会任务格式、输出结构和基本排错风格
2. 再做 DPO，强化“最小修复、少幻觉、信息不足先追问”这些偏好

默认配置已经指向最终训练集：

- [sft_qwen3_4b_lora.json](/D:/LLM_Learning/SFT/configs/sft_qwen3_4b_lora.json)
- [dpo_qwen3_4b_lora.json](/D:/LLM_Learning/SFT/configs/dpo_qwen3_4b_lora.json)

## 本地准备

如果本地不训练，只需要做两件事：

1. 校验数据文件结构
2. 校验训练脚本能否正常读取数据并完成 prompt 格式化

先安装依赖：

```bash
pip install -r requirements.txt
```

再执行：

```bash
python validate_dataset.py data/final/processed/sft_full_v1.jsonl
python validate_dataset.py data/final/processed/dpo_full_v1.jsonl
python validate_dataset.py data/train_data/data_v1.json
```

如果本地没有 GPU，可以只做 sanity check：

```bash
python train_sft.py --config_path configs/sft_qwen3_4b_lora.json --sanity_check_only true
python train_dpo.py --config_path configs/dpo_qwen3_4b_lora.json --sanity_check_only true
```

## 云端训练

云端拉取仓库后，直接按下面顺序执行即可：

```bash
pip install -r requirements.txt
bash run_sft.sh
bash run_dpo.sh
```

如果需要指定配置文件：

```bash
bash run_sft.sh configs/sft_qwen3_4b_lora.json
bash run_dpo.sh configs/dpo_qwen3_4b_lora.json
```

默认是面向 `Qwen/Qwen3-4B-Instruct-2507` 的 LoRA 训练配置。如果云端显存吃紧，可以把配置里的 `use_4bit` 改为 `true`，切换到 QLoRA 路线。

## benchmark 评估

基准集放在：

- [data_v1.json](/D:/LLM_Learning/SFT/data/train_data/data_v1.json)

这份数据更适合拿来做开发期评估，而不是直接混进训练集。里面包含：

- 常见运行时异常
- 逻辑错误
- 信息不足样本
- 报错和代码不匹配样本

当前仓库里有两层评测数据：

- [data_v1.json](/D:/LLM_Learning/SFT/data/train_data/data_v1.json)：完整 benchmark 种子集
- [benchmark_dev_v1.json](/D:/LLM_Learning/SFT/data/benchmark/benchmark_dev_v1.json) 与 [benchmark_test_v1.json](/D:/LLM_Learning/SFT/data/benchmark/benchmark_test_v1.json)：已经拆好的开发集和测试集

如果云端推理后导出一个预测文件，例如：

```json
{"id": "test_001", "response": "...模型完整回答..."}
```

可以直接跑离线评估：

```bash
python evaluate.py --benchmark-path data/train_data/data_v1.json --predictions-path predictions.jsonl
```

当前评估脚本会统计：

- 样本覆盖率
- 严格关键点命中率
- 宽松关键点命中率
- `error_type` 准确率
- JSON 合法率
- `need_more_context` 准确率

如果你想导出逐样本明细，方便对比 `base / sft / dpo` 在哪些题上出现分歧，可以额外传入：

```bash
python evaluate.py \
  --benchmark-path data/benchmark/benchmark_dev_v1.json \
  --predictions-path output/predictions_dpo_dev.jsonl \
  --report-path output/eval_report_dpo_dev.jsonl
```

## baseline 对比

为了判断微调是否真的带来了收益，建议至少比较三组结果：

1. `base`：原始 `Qwen/Qwen3-4B-Instruct-2507`
2. `sft`：SFT 产物
3. `dpo`：DPO 产物

推理脚本在：

- [run_benchmark_inference.py](/D:/LLM_Learning/SFT/run_benchmark_inference.py)

示例用法：

```bash
python run_benchmark_inference.py \
  --mode base \
  --benchmark-path data/benchmark/benchmark_dev_v1.json \
  --predictions-path output/predictions_base_dev.jsonl

python run_benchmark_inference.py \
  --mode sft \
  --benchmark-path data/benchmark/benchmark_dev_v1.json \
  --predictions-path output/predictions_sft_dev.jsonl

python run_benchmark_inference.py \
  --mode dpo \
  --benchmark-path data/benchmark/benchmark_dev_v1.json \
  --predictions-path output/predictions_dpo_dev.jsonl
```

然后分别评估：

```bash
python evaluate.py --benchmark-path data/benchmark/benchmark_dev_v1.json --predictions-path output/predictions_base_dev.jsonl
python evaluate.py --benchmark-path data/benchmark/benchmark_dev_v1.json --predictions-path output/predictions_sft_dev.jsonl
python evaluate.py --benchmark-path data/benchmark/benchmark_dev_v1.json --predictions-path output/predictions_dpo_dev.jsonl
```

如果你只想做最终结果对比，可以把 `benchmark_dev_v1.json` 换成 `benchmark_test_v1.json`。

## 训练过程可视化

训练完成后，`Trainer` 会在输出目录中留下 `trainer_state.json`。可以直接用下面的脚本把 `loss / eval_loss / learning_rate` 画出来：

```bash
python plot_training_curves.py --output-dir output/qwen3_sft_lora
python plot_training_curves.py --output-dir output/qwen3_dpo_lora
```

默认会在对应输出目录下生成：

- `training_curves.png`

如果你想把两次实验放在同一张图里对比，例如对比 SFT 和 DPO，或者对比两组超参数，可以直接这样用：

```bash
python plot_training_curves.py \
  --compare-dirs output/qwen3_sft_lora output/qwen3_dpo_lora \
  --labels SFT DPO \
  --save-path output/training_curves_compare.png
```

对比模式会把多次实验的 `loss / eval_loss / learning_rate` 叠加到同一张图上，便于观察收敛速度和验证集变化。

## 备注

这个项目最重要的不是把训练流程堆复杂，而是把任务边界守住。对于 4B 量级的小模型来说，数据风格统一、错误类型覆盖合理、偏好定义明确，通常比盲目继续加参数更重要。
