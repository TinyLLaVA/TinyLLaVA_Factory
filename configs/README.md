# Configuration

Run from the repository root. Configuration has three reusable parts:

| Directory | Contents |
| --- | --- |
| `models/` | Language model, vision tower, connector, tokenizer and preprocessing |
| `data/` | Named training datasets; `data/eval/` holds benchmark inputs |
| `recipes/` | Training policies and optimizer defaults |
| `experiments/` | Model selection and ordered training stages |
| `train/` | Minimal single-stage examples |
| `eval/` | Generation settings for each benchmark |
| `deepspeed/` | ZeRO configuration |

For example, `experiments/phi.yaml` is:

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

```bash
python -m tinyllava.run --config configs/experiments/phi.yaml --dry-run
python -m tinyllava.run --config configs/experiments/phi.yaml
python -m tinyllava.run --config configs/experiments/phi.yaml --steps finetune
```

`pretrain` and `finetune` are arbitrary stage names. Renaming them to `align` and
`sft` changes output directories, not training behavior. `recipe` chooses defaults;
`training.tune_type_*`, the dataset and model options determine actual behavior.
A renamed stage still needs its intended recipe.

Outputs default to `output/<experiment>/<stage>/`; the experiment name defaults
to the experiment file's stem. The next stage loads the previous stage's complete
checkpoint. No output variables or checkpoint paths are needed. Selecting only a
later stage still infers the earlier stage's output path.

```bash
python -m tinyllava.run --config configs/experiments/phi.yaml \
  name=phi_trial dataset_dir=/datasets/llava \
  stages.finetune.training.learning_rate=1e-5
```

- `name`: name an independent run. Stable names support resuming existing jobs.
- `output_root`: move all outputs together (default `output`).
- `stages.NAME.training.output_dir`: optional explicit directory; downstream stages follow it.
- `stages.NAME.from`: select an earlier stage instead of the immediately previous one.
- `stages.NAME.checkpoint`: initialize from an external checkpoint; `null` starts from components.
- `model=gemma`: choose a different model preset. `model.model_max_length=4096` changes a model field.
- `dataset_dir`: shared data root, default `datasets` or `TINYLLAVA_DATASET_DIR`.

## Single-stage training

```yaml
model: qwen2_instruct
recipe: pretrain
dataset: llava_558k
```

Use `configs/train/pretrain.yaml` or the corresponding fine-tuning/LoRA/QLoRA file:

```bash
python -m tinyllava.run --config configs/train/pretrain.yaml model=phi
python -m tinyllava.train.train --config configs/train/finetune.yaml model=phi
```

Single-stage output defaults to `output/<model preset>/<config filename>/`.
`stage=align` changes the last component; `name=my-run` changes the experiment
component. Fine-tuning examples use `from: pretrain`, which infers the checkpoint
at `output/<experiment>/pretrain`. Set `checkpoint=/path/to/checkpoint` to use an
existing checkpoint elsewhere. Expanded `model`, `data`, `training`, `peft` mappings
remain supported. Model and data field overrides are merged before interpolation.

## Distributed training

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m tinyllava.run \
  --config configs/experiments/qwen2_base_legacy.yaml
```

The legacy recipe enables torchrun and preserves global batch sizes 256/128.
For another experiment, set `launcher.distributed=true`; `launcher.devices=auto`
uses `CUDA_VISIBLE_DEVICES` (default GPU 0). A list or comma-separated GPU string
also works. Set `stages.NAME.launcher.global_batch_size` to derive accumulation
from GPU count and per-device batch; otherwise the recipe's fixed accumulation is
used. DeepSpeed settings live in `configs/deepspeed/`.

## Evaluation

```bash
python -m tinyllava.run --config configs/eval/textvqa.yaml \
  model=output/phi/finetune runtime.batch_size=4
python -m tinyllava.run --config configs/eval/gqa.yaml \
  model=output/phi/finetune devices=0,1 --dry-run
```

Inputs default to `datasets/eval/`. Outputs default to
`output/phi/finetune/eval/<benchmark>/` for that checkpoint. An external checkpoint
uses `output/<checkpoint basename>/eval/<benchmark>/`; use `output_dir` to choose
another result directory. Model IDs and `MODEL_PATH`, `MODEL_NAME`, `EVAL_DIR`
environment variables remain supported. `eval_dir` overrides the evaluation data
root without changing training data. Results are kept outside dataset directories.

Use `--steps generate` for generation only, or call
`python -m tinyllava.eval.batch_generation --config configs/eval/textvqa.yaml`.
GQA and VQAv2 shard over visible GPUs and merge only after all workers succeed.
MME uses a private workspace under its output directory because its official
converter expects fixed relative paths. Official GQA/MME/MMMU evaluators remain
part of the downloaded evaluation assets. VQAv2 and MM-Vet export for external scoring.

Legacy Qwen2 ScienceQA uses `configs/eval/scienceqa_qwen2_base_legacy.yaml`.
For other benchmarks, set
`generation.chat_template_path=configs/chat_templates/qwen2_base_legacy.jinja`.

`--dry-run` prints resolved configs and commands without loading models. It does
not validate model downloads, dataset completeness or checkpoint completeness.

The dataset-name convention follows [LLaMA-Factory's dataset registry](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md).
Automatic output and stage checkpoint inference are TinyLLaVA conventions;
LLaMA-Factory's examples specify `output_dir` explicitly.
