# 训练

配置分为模型（`configs/models/`）、数据集（`configs/data/`）、训练策略（`configs/recipes/`）和实验（`configs/experiments/`）。从仓库根目录运行：

```bash
python -m tinyllava.run --config configs/experiments/phi.yaml --dry-run
python -m tinyllava.run --config configs/experiments/phi.yaml
```

一个实验只需指定：

```yaml
model: phi
stages:
  pretrain:
    recipe: pretrain
    dataset: llava_558k
  finetune:
    recipe: finetune
    dataset: llava_665k
```

`pretrain`、`finetune` 是自由的阶段名，可以改成 `align`、`sft`。训练行为由 `recipe`、数据和实际参数决定，不根据阶段名字判断。默认输出为 `output/<实验名>/<阶段名>/`，实验名默认取配置文件名。后续阶段自动加载上一阶段完整 checkpoint，保留已训练的连接器。

```bash
python -m tinyllava.run --config configs/experiments/phi.yaml \
  name=phi_trial stages.finetune.training.learning_rate=1e-5
python -m tinyllava.run --config configs/experiments/phi.yaml --steps finetune
```

选择单个后续阶段时，仍按完整实验推断前序 checkpoint。`stages.sft.from=align` 可以指定更早阶段；`stages.sft.checkpoint=/path/to/model` 可以加载外部 checkpoint，设为 `null` 则重新从基础组件组装。更改某阶段的 `training.output_dir` 后，下游加载路径自动跟随。`output_root` 可以整体移动输出根目录。

## 单阶段与自定义数据

```bash
python -m tinyllava.run --config configs/train/pretrain.yaml model=phi
python -m tinyllava.train.train --config configs/train/finetune.yaml \
  model=phi checkpoint=/path/to/pretrained-model
```

单阶段示例默认使用 `output/<模型名>/<配置文件名>/`，`name` 和 `stage` 可分别修改这两部分。`from: pretrain` 表示加载同一实验目录下的 `pretrain`。LoRA、QLoRA 示例分别位于 `configs/train/lora_finetune.yaml` 和 `qlora_finetune.yaml`。

数据默认位于 `datasets/`，用 `dataset_dir=/path/to/data` 整体切换根目录。新增数据集时，在 `configs/data/` 定义标注和图片路径，再用 `dataset=名称` 选择。临时覆盖仍支持 `data.dataset_name_or_path`、`data.image_folder`。详见[数据指南](data.md)。

## 多卡与恢复

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m tinyllava.run \
  --config configs/experiments/qwen2_base_legacy.yaml
```

legacy 配方使用 Qwen2 base 和论文提示词，自动保持预训练全局 batch 256、微调 128。其他实验用 `launcher.distributed=true` 启用 torchrun；`launcher.devices` 默认读取 `CUDA_VISIBLE_DEVICES`。通过 `stages.阶段.launcher.global_batch_size` 设置目标全局 batch，自动换算梯度累积；不设置时保留策略中的固定值。DeepSpeed 配置位于 `configs/deepspeed/`。

`training.resume_from_checkpoint=auto` 恢复当前输出目录中的完整 checkpoint；阶段间加载模型用于初始化新阶段。独立实验应设置新的 `name`。`--dry-run` 只解析配置并打印命令，不检查数据完整性或加载模型。
