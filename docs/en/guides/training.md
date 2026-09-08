# Training

Training YAML has three sections: `model`, `data`, and `training`.
The parser accepts OmegaConf dotlist overrides; run from the repository root
so relative config and checkpoint paths resolve consistently.

```bash
python -m tinyllava.train.train \
  --config configs/train/models/qwen2_base_pretrain.yaml \
  data.data_path=/path/to/blip_laion_cc_sbu_558k.json \
  data.image_folder=/path/to/pretrain/images
```

## Two stages and checkpoint continuity

Pretraining assembles components from `language_model_name_or_path`,
`vision_model_name_or_path`, and `connector_config`. Fine-tuning must load the
result using `model.pretrained_model_name_or_path`. Reassembling the components
from their base model IDs would discard the trained connector.

```bash
python -m tinyllava.train.train \
  --config configs/train/models/qwen2_base_finetune.yaml \
  model.pretrained_model_name_or_path=output/tinyllava-qwen2-base-pretrain \
  data.data_path=/path/to/llava_v1_5_mix665k.json \
  data.image_folder=/path/to/dataset
```

## Distributed training and global batch

```text
global batch = GPU count × per-device batch × gradient accumulation
```

The generic YAML files have fixed batch values; changing GPU count requires
adjusting accumulation yourself. For example, fine-tuning with four GPUs,
micro batch 4, and accumulation 8 gives a global batch of 128:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  --module tinyllava.train.train \
  --config configs/train/models/qwen2_base_finetune.yaml \
  training.gradient_accumulation_steps=8 \
  data.data_path=/path/to/llava_v1_5_mix665k.json \
  data.image_folder=/path/to/dataset
```

## Qwen2 base with the paper's legacy prompts

The dedicated launcher adjusts accumulation to preserve pretraining batch 256
and fine-tuning batch 128. Set local data paths explicitly:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TINYLLAVA_PRETRAIN_DATA=/path/to/blip_laion_cc_sbu_558k.json
export TINYLLAVA_PRETRAIN_IMAGE_FOLDER=/path/to/pretrain/images
export TINYLLAVA_FINETUNE_DATA=/path/to/llava_v1_5_mix665k.json
export TINYLLAVA_FINETUNE_IMAGE_FOLDER=/path/to/dataset
bash scripts/train/qwen2/train_qwen2_base_legacy.sh check
bash scripts/train/qwen2/train_qwen2_base_legacy.sh all
```

This recipe uses the base model `Qwen/Qwen2-0.5B`. It selects
`pretrain_legacy.jinja` for caption alignment and `qwen2_base_legacy.jinja`
for fine-tuning.

## Resume and tuning policy

`training.resume_from_checkpoint=auto` discovers complete checkpoints in the
output directory. An existing resumable checkpoint continues optimizer state;
loading `model.pretrained_model_name_or_path` instead initializes a new stage.
Use a new output directory for an independent experiment.

Tune policies are controlled by `tune_type_llm`, `tune_type_vision_tower`, and
`tune_type_connector`. LoRA/QLoRA examples are under `configs/train/` and use
`training.peft_config`. See the [training API](../reference/training.md).

`training.group_by_modality_length` is optional and defaults to false. It groups
text-only and image samples. Compare throughput and padding on your dataset
when choosing whether to enable it.
