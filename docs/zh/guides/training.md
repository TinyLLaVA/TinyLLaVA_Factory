# 训练

训练 YAML 分为 `model`、`data`、`training`。命令应在仓库根目录运行；OmegaConf dotlist 参数可以覆盖 YAML 字段。

```bash
python -m tinyllava.train.train \
  --config configs/train/models/qwen2_base_pretrain.yaml \
  data.data_path=/path/to/blip_laion_cc_sbu_558k.json \
  data.image_folder=/path/to/pretrain/images
```

微调必须加载预训练结果，重新从基础组件组装会丢失已训练的连接器：

```bash
python -m tinyllava.train.train \
  --config configs/train/models/qwen2_base_finetune.yaml \
  model.pretrained_model_name_or_path=output/tinyllava-qwen2-base-pretrain \
  data.data_path=/path/to/llava_v1_5_mix665k.json \
  data.image_folder=/path/to/dataset
```

## 多卡与全局 batch

全局 batch = GPU 数 × 每卡 batch × 梯度累积步数。更改 GPU 数时，需同步调整通用 YAML 中的梯度累积。四卡、每卡 4、累积 8 的全局 batch 为 128：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  --module tinyllava.train.train \
  --config configs/train/models/qwen2_base_finetune.yaml \
  training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=8 \
  data.data_path=/path/to/llava_v1_5_mix665k.json \
  data.image_folder=/path/to/dataset
```

## Qwen2 legacy 配方

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TINYLLAVA_PRETRAIN_DATA=/path/to/blip_laion_cc_sbu_558k.json
export TINYLLAVA_PRETRAIN_IMAGE_FOLDER=/path/to/pretrain/images
export TINYLLAVA_FINETUNE_DATA=/path/to/llava_v1_5_mix665k.json
export TINYLLAVA_FINETUNE_IMAGE_FOLDER=/path/to/dataset
bash scripts/train/qwen2/train_qwen2_base_legacy.sh check
bash scripts/train/qwen2/train_qwen2_base_legacy.sh all
```

该启动器保持预训练 batch 256、微调 batch 128；使用基础模型 `Qwen/Qwen2-0.5B`，预训练选择 `pretrain_legacy.jinja`，微调选择 `qwen2_base_legacy.jinja`。

## 恢复与调优

`training.resume_from_checkpoint=auto` 查找完整 checkpoint 并恢复训练状态；`model.pretrained_model_name_or_path` 则用于初始化一个新阶段。独立实验应使用新输出目录。调优策略见 `tune_type_llm`、`tune_type_vision_tower`、`tune_type_connector` 和 `training.peft_config`。

`training.group_by_modality_length` 默认关闭，启用后按模态与长度分组采样。可比较实际数据上的吞吐和 padding 开销，再决定是否启用。
