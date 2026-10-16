# 训练

配置分为模型（`configs/models/`）、数据集（`configs/data/train/`）、训练策略（`configs/training/`）和实验（`configs/experiments/`）。预设只包含所属节点的字段，分别由 `model.from`、`data.from`、`training.from` 引用；评测数据独立放在 `configs/data/eval/`。从仓库根目录运行：

```bash
python -m tinyllava.cli train --config configs/experiments/phi.yaml --dry-run
python -m tinyllava.cli train --config configs/experiments/phi.yaml
```

一个实验只需指定：

```yaml
# configs/experiments/phi.yaml
algorithm: sft
model: phi
stages:
  pretrain:
    data: llava_558k
    training: pretrain
  finetune:
    data: llava_665k
    training: finetune
```

文件名在对应预设目录查找；只引用文件时可写 `model: phi`，需要覆写时写成
`model: {from: phi, model_max_length: 4096}`。显式相对路径以声明它的 YAML 为基准；
命令行可写 `model=phi model.model_max_length=4096`。详见[配置参考](../reference/configuration.md)。

`pretrain`、`finetune` 是自由的阶段名，可以改成 `align`、`sft`。训练行为由训练配置、数据和实际参数决定，不根据阶段名字判断。默认输出为 `output/<实验名>/<阶段名>/`，实验名默认取配置文件名。后续阶段自动加载上一阶段完整 checkpoint，保留已训练的连接器。

```bash
python -m tinyllava.cli train --config configs/experiments/phi.yaml \
  experiment_name=phi_trial stages.finetune.training.learning_rate=1e-5
python -m tinyllava.cli train --config configs/experiments/phi.yaml --steps finetune
```

选择单个后续阶段时，仍按完整实验推断前序 checkpoint。`stages.sft.from_stage=align` 可以指定更早阶段；`stages.sft.checkpoint=/path/to/model` 可以加载外部 checkpoint，设为 `null` 则重新从基础组件组装。更改某阶段的 `training.saving.output_dir` 后，下游加载路径自动跟随。`output_root` 可以整体移动输出根目录。

## 单阶段与自定义数据

从实验中选择要运行的阶段：

```bash
python -m tinyllava.cli train --config configs/experiments/phi.yaml --steps pretrain
python -m tinyllava.cli train --config configs/experiments/phi_lora.yaml --steps finetune \
  experiment_name=phi_custom stages.finetune.checkpoint=/path/to/pretrained-model
```

`configs/experiments/` 统一作为训练入口，`--steps` 选择执行阶段并保留 checkpoint 依赖。
LoRA、QLoRA 示例为 `phi_lora.yaml`、`phi_qlora.yaml`；训练预设不再强制长度和注意力后端。
`model_max_length` 控制截断长度，`attn_implementation` 选择注意力实现，均不是 LoRA 的必要条件。
按模型、数据和环境需要，可显式覆写 `stages.finetune.model.model_max_length=3072` 或
`stages.finetune.model.attn_implementation=flash_attention_2`。

自定义单阶段入口可以省略 `stages`：

```yaml
# configs/experiments/custom.yaml
algorithm: sft
stage: finetune
checkpoint: /path/to/pretrained-model
model: phi
data: llava_665k
training:
  from: lora_finetune
  learning_rate: 0.0001
```

使用 `--config configs/experiments/custom.yaml` 运行，默认输出为 `output/custom/finetune/`。
`experiment_name` 和 `stage` 可分别修改目录的这两部分；`from_stage: pretrain` 可加载同一实验下的预训练输出。

数据默认位于 `datasets/`，用 `dataset_dir=/path/to/data` 整体切换根目录。
在 `configs/data/train/` 定义新数据集后，通过 `stages.finetune.data.from=名称` 选择；
也可以直接覆写 `stages.finetune.data.dataset_name_or_path`、`stages.finetune.data.image_folder`。
单阶段入口使用 `data.*`。详见[数据指南](data.md)。

## 多卡与恢复

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m tinyllava.cli train \
  --config configs/experiments/qwen2_base_legacy.yaml
```

legacy 配方使用 Qwen2 base 和论文提示词，自动保持预训练全局 batch 256、微调 128。其他实验用 `launcher.distributed=true` 启用 torchrun；`launcher.devices` 默认读取 `CUDA_VISIBLE_DEVICES`。通过 `stages.阶段.training.global_batch_size` 设置目标全局 batch，自动换算梯度累积；不设置时保留策略中的固定值。DeepSpeed 配置位于 `configs/deepspeed/`。

`training.saving.resume_from_checkpoint=auto` 恢复当前输出目录中的完整 checkpoint；阶段间加载模型用于初始化新阶段。独立实验应设置新的 `experiment_name`。`--dry-run` 只解析配置并打印命令，不检查数据完整性或加载模型。
