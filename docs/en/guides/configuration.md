# Configuration

Choose a YAML entry with `--config`, then override individual values with `key=value`.
Use the [parameter reference](parameters.md) to look up fields and the
[model guide](models.md) to choose components or a checkpoint.

## Training stages

```yaml
# configs/experiments/custom.yaml
model: phi

stages:
  pretrain:
    algorithm: sft
    data: llava_558k
    training: pretrain

  finetune:
    algorithm: sft
    data: llava_665k
    training:
      from: lora_finetune
      learning_rate: 0.0001
```

Each stage selects its own `algorithm` and `data`. The root can share `model`,
`training`, and `launcher` settings. Stage values override shared values, including
values from referenced presets. A single-stage entry omits `stages` and places
its algorithm, data, model, and training settings at the root.

Trainer fields go directly under `training`, such as `save_steps` and
`logging_steps`. Component update policies belong to `training.tuning`;
LoRA options belong to `training.tuning.peft`.

## Reuse presets

`from` loads a YAML file into the current node. Local values override the file:

```yaml
model:
  from: phi
  model_max_length: 4096
  connector:
    from: mlp2x_gelu
    depth: 3
```

When a node only selects a file, `model: phi` is equivalent to
`model: {from: phi}`. Bare names resolve in these directories:

| Node | Preset directory |
| --- | --- |
| `model` | `configs/models/` |
| `connector` | `configs/connectors/` |
| `training` | `configs/training/` |
| `data` | `configs/data/train/` |

The same lookup applies inside stages and to entry files outside the repository.
Omitting the extension adds `.yaml`; explicit `.yaml` and `.yml` are accepted.
Use `./custom.yaml` to select a file beside the declaring YAML.
Paths containing directories resolve relative to that YAML; absolute paths and
`~` are supported. Explicit relative paths on the command line use the working directory.
Checkpoint, annotation, image, and template paths remain ordinary field values.

To combine files, use a list: `training: {from: [pretrain, ./hardware]}`.
Files merge left to right, followed by local fields and command-line overrides.
Mappings merge recursively; lists and scalar values are replaced.
Changing a reference preserves explicit sibling overrides and replaces the old
preset's defaults. `from` also works at the root and in mappings inside lists.

## Evaluation entries

Evaluation fields live at the root. Reuse benchmark input definitions with a root reference:

```yaml
# configs/eval/custom.yaml
from: ../data/eval/gqa.yaml
benchmark: gqa

model_name_or_path: output/phi/finetune
temperature: 0.0
batch_size: 4
output_dir: output/custom-eval
```

`output_dir` selects the result directory; `answers_file` selects the prediction
file and defaults to `${output_dir}/answers.jsonl` in the evaluation workflow.
See [Evaluation](evaluation.md) for dataset preparation, sharding, and scoring.

## Override and preview

```bash
python -m tinyllava.cli train --config configs/experiments/phi.yaml \
  model=phi model.model_max_length=4096 \
  stages.finetune.training.learning_rate=1e-5 --dry-run

python -m tinyllava.cli eval --config configs/eval/gqa.yaml \
  model_name_or_path=output/phi/finetune batch_size=4 --dry-run
```

`--dry-run` prints resolved settings and commands. Use `--steps finetune` to select
a training stage, or `--steps generate` to generate benchmark answers. Training
stage selection preserves checkpoint dependencies; see [Training](training.md).

Values support OmegaConf interpolation, for example `${dataset_dir}/images` and
`${oc.env:EVAL_CHAT_TEMPLATE_PATH,null}`. References in `from` use literal paths.
For the Python configuration interfaces, see the [API reference](../reference/configuration.md).
