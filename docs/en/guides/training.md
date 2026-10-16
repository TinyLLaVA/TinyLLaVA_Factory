# Training

Model definitions live in `configs/models/`, dataset definitions in `configs/data/train/`,
and training settings in `configs/training/`. Each preset contains only the fields
of its target node: `model.from`, `data.from`, or `training.from`. Evaluation data
is separate in `configs/data/eval/`. An experiment combines them:

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

Bare filenames select the matching preset directory. A node with only a file
reference can use `model: phi`; use `model: {from: phi, model_max_length: 4096}`
when overriding fields. Explicit relative paths are anchored to the declaring YAML.
CLI overrides support `model=phi model.model_max_length=4096`. See the [configuration reference](../reference/configuration.md).

`model.vision_feature_select_strategy` is inferred from the vision backbone by
default: SigLIP retains every patch, while CLIP, DINO and MoF drop the leading CLS
token. Set `full` or `default` explicitly to override this behavior. Loading a
checkpoint without an override preserves its saved strategy.

Run from the repository root:

```bash
python -m tinyllava.cli train --config configs/experiments/phi.yaml --dry-run
python -m tinyllava.cli train --config configs/experiments/phi.yaml
python -m tinyllava.cli train --config configs/experiments/phi.yaml --steps finetune
```

Stage names are arbitrary: `align` and `sft` work equally well. The selected training settings and
actual model/data options determine behavior. Outputs default to
`output/<experiment>/<stage>/`; the experiment name defaults to the file stem.
Each later stage loads the previous stage's complete checkpoint, preserving the
trained connector. Selecting only a later stage still infers its input checkpoint.

```bash
python -m tinyllava.cli train --config configs/experiments/phi.yaml \
  experiment_name=phi_trial stages.finetune.training.learning_rate=1e-5
```

Use `stages.sft.from_stage=align` to load an earlier stage, or
`stages.sft.checkpoint=/path/to/model` for an external checkpoint. A null checkpoint
starts from base components. An explicit stage `training.saving.output_dir` is respected
and downstream inputs follow it. `output_root` relocates all default output paths.

## Single-stage training and custom data

Select a stage from an experiment:

```bash
python -m tinyllava.cli train --config configs/experiments/phi.yaml --steps pretrain
python -m tinyllava.cli train --config configs/experiments/phi_lora.yaml --steps finetune \
  experiment_name=phi_custom stages.finetune.checkpoint=/path/to/pretrained-model
```

`configs/experiments/` is the training entry directory. `--steps` chooses execution
while preserving checkpoint dependencies. LoRA/QLoRA examples are `phi_lora.yaml`
and `phi_qlora.yaml`. Their training presets do not impose a sequence length or
attention backend: these remain model settings. For a specific run, set
`stages.finetune.model.model_max_length=3072` or
`stages.finetune.model.attn_implementation=flash_attention_2` when appropriate for
the model and environment. Neither setting is a LoRA requirement.

A custom single-stage entry can omit `stages`:

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

Run this file with `--config configs/experiments/custom.yaml`. Its output defaults
to `output/custom/finetune/`; `experiment_name` and `stage` override those components.
`from_stage: pretrain` can select the pretraining output under the same experiment.

Inputs default to `datasets/`. Change the root with `dataset_dir=/path/to/data`.
Define custom datasets under `configs/data/train/`, then select
`stages.finetune.data.from=NAME`, or supply
`stages.finetune.data.dataset_name_or_path` and `stages.finetune.data.image_folder`
directly. Single-stage entries use `data.*`. See the [data guide](data.md).

## Distributed training and resuming

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m tinyllava.cli train \
  --config configs/experiments/qwen2_base_legacy.yaml
```

The legacy Qwen2 experiment preserves global batches 256/128. Other experiments
can set `launcher.distributed=true` for torchrun; devices default to
`CUDA_VISIBLE_DEVICES`. Set `stages.NAME.training.global_batch_size` to derive
accumulation from GPU count and per-device batch. Otherwise the preset accumulation values
are retained. DeepSpeed configuration lives in `configs/deepspeed/`.

`training.saving.resume_from_checkpoint=auto` resumes complete checkpoints in the current
output directory. Loading a previous stage's model initializes a new stage.
Choose a new `experiment_name` for an independent run. `--dry-run` prints resolved parameters
and commands without loading models or checking dataset/checkpoint completeness.
