# Training

Model definitions live in `configs/models/`, dataset definitions in `configs/data/`,
and training policies in `configs/recipes/`. An experiment combines them:

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

Run from the repository root:

```bash
python -m tinyllava.run --config configs/experiments/phi.yaml --dry-run
python -m tinyllava.run --config configs/experiments/phi.yaml
python -m tinyllava.run --config configs/experiments/phi.yaml --steps finetune
```

Stage names are arbitrary: `align` and `sft` work equally well. The recipe and
actual model/training/data options determine behavior. Outputs default to
`output/<experiment>/<stage>/`; the experiment name defaults to the file stem.
Each later stage loads the previous stage's complete checkpoint, preserving the
trained connector. Selecting only a later stage still infers its input checkpoint.

```bash
python -m tinyllava.run --config configs/experiments/phi.yaml \
  name=phi_trial stages.finetune.training.learning_rate=1e-5
```

Use `stages.sft.from=align` to load an earlier stage, or
`stages.sft.checkpoint=/path/to/model` for an external checkpoint. A null checkpoint
starts from base components. An explicit stage `training.output_dir` is respected
and downstream inputs follow it. `output_root` relocates all default output paths.

## Single-stage training

```bash
python -m tinyllava.run --config configs/train/pretrain.yaml model=phi
python -m tinyllava.train.train --config configs/train/finetune.yaml \
  model=phi checkpoint=/path/to/pretrained-model
```

Single-stage defaults are `output/<model preset>/<config filename>/`; `name` and
`stage` override those components. `from: pretrain` selects the pretraining output
under that experiment. LoRA/QLoRA examples are `configs/train/lora_finetune.yaml`
and `qlora_finetune.yaml`.

Inputs default to `datasets/`. Change the root with `dataset_dir=/path/to/data`.
Register custom datasets under `configs/data/`, then select `dataset=NAME`, or
supply `data.dataset_name_or_path` and `data.image_folder` directly. See the
[data guide](data.md).

## Distributed training and resuming

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m tinyllava.run \
  --config configs/experiments/qwen2_base_legacy.yaml
```

The legacy Qwen2 experiment preserves global batches 256/128. Other experiments
can set `launcher.distributed=true` for torchrun; devices default to
`CUDA_VISIBLE_DEVICES`. Set `stages.NAME.launcher.global_batch_size` to derive
accumulation from GPU count and per-device batch. Otherwise fixed recipe values
are retained. DeepSpeed configuration lives in `configs/deepspeed/`.

`training.resume_from_checkpoint=auto` resumes complete checkpoints in the current
output directory. Loading a previous stage's model initializes a new stage.
Choose a new `name` for an independent run. `--dry-run` prints resolved parameters
and commands without loading models or checking dataset/checkpoint completeness.
