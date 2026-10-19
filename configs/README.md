# Configuration

TinyLLaVA provides one YAML composition feature: `from`. At any mapping node,
reference a YAML file and place overrides beside it. Preset names can omit both their
directory and `.yaml`; a node that only references a file can use its name directly:

```yaml
model: phi
data: llava_558k
training: pretrain
```

These are equivalent to `model: {from: phi}`, `model: phi.yaml`, etc. To add overrides, expand
the corresponding node:

```yaml
# configs/experiments/custom.yaml
algorithm: sft
experiment_name: phi_trial
model:
  from: phi
  model_max_length: 4096
  connector:
    from: mlp2x_gelu
    depth: 3
data:
  from: llava_558k
training:
  from: pretrain
  learning_rate: 0.001
```

`model.from` loads the file's contents into `model`. That file can reference its
connector in exactly the same way. A root `from` merges a whole configuration.
Each referenced file must contain a mapping, matching the node where it is used.
Inline settings work without a file reference too.

## Composition rules

- `from: base` loads one file; `from: [base.yaml, local.yaml]` merges several
  files in order. Later files win, then local fields, then CLI overrides.
- Mappings merge recursively; lists and scalar values are replaced. References
  work at any depth, including mappings inside lists.
- Bare filenames at preset nodes use the directories in the table below. They
  never fall back to a local file with the same name. Omitting the extension appends `.yaml`; explicit `.yaml` and `.yml` still work.
- `./custom` (or `./custom.yaml`) and `../custom` explicitly refer to files relative to the
  declaring YAML; explicit CLI paths are relative to the working directory.
  Absolute paths and `~` work too. At nodes without a preset directory, `from`
  references retain the relative-file rule.
- Direct preset-name shorthand applies to `model`, `connector`, `training`, and
  `data`. Ordinary data, template, output, and checkpoint path fields remain values.
- `from` takes literal paths. Value interpolation such as `${dataset_dir}` and
  `${oc.env:NAME,default}` remains available through OmegaConf, and is resolved
  after the workflow has assigned runtime paths.
- Changing a `from` selects a new source before nested references are expanded.
  The new preset replaces the previous defaults and preserves explicit sibling overrides.
- `from` is reserved for composition and is removed from the result. Missing files,
  non-mapping files, malformed references, and cycles report configuration errors.
  Reusing a file in different places is supported.

| Node | Directory for bare filenames |
| --- | --- |
| `model` | `configs/models/` |
| `connector`, including `model.connector` | `configs/connectors/` |
| `training`, including stage training | `configs/training/` |
| `data` in a training entry | `configs/data/train/` |

These directories also apply to entries outside the repository. Evaluation
parameters are flat; merge benchmark inputs from `configs/data/eval/` at the root
with an explicit relative or absolute `from` reference.

Preset files contain the fields of their destination node. `training/pretrain.yaml`
starts with `learning_rate`, `tuning`, etc.; it has no outer `training` mapping.
The same rule applies to model, connector, training-data and evaluation-data files.
For example, combine training settings with:

```yaml
training:
  from: [pretrain, ./hardware]
  learning_rate: 0.001
```

## Overrides

Run commands from the repository root. The same `key=value` syntax adds or replaces
fields, including file selections:

```bash
python -m tinyllava.cli train --config configs/experiments/qwen2_instruct.yaml --steps pretrain \
  model=phi stages.pretrain.data=llava_665k \
  model.model_max_length=4096 experiment_name=phi_trial --dry-run

python -m tinyllava.cli train --config configs/experiments/qwen2_instruct.yaml --steps pretrain \
  model.connector=mof \
  model.connector.clip_config.depth=4 --dry-run
```

Shorthand and nested fields work together, such as
`model=phi model.model_max_length=4096`. Shorthand selects the reference
and preserves explicit sibling overrides, just like `model.from=phi`.

Use `launcher.devices=[0,1]` for a device list. Unknown application fields are
checked by the workflow and algorithm after composition.

## Multiple stages

```yaml
# configs/experiments/custom.yaml
model: phi
stages:
  align:
    algorithm: sft
    data: llava_558k
    training: pretrain
  instruction:
    algorithm: sft
    data: llava_665k
    training: finetune
```

The configuration layer composes files first. The workflow then merges shared
model, training, and launcher defaults with each stage; stage fields win, including
values from training presets. Each stage selects its own `algorithm` and `data`.
Override a stage with `stages.align.training.learning_rate=...`, select its training preset
with `stages.align.training.from=pretrain`, or select its data with
`stages.align.data.from=llava_558k`. Stage names are independent
of the training algorithm.

Outputs default to `output/<experiment_name or entry filename>/<stage>/`.
Each stage initializes from the previous output. `stages.NAME.from_stage` selects
an earlier stage; `checkpoint` selects an external checkpoint, and `checkpoint: null`
starts from components. `training.resume_from_checkpoint` instead resumes
an interrupted run in its output directory. `--steps NAME` selects execution,
while dependencies are still resolved using the whole experiment.

## File organization and ownership

| Directory | Contents |
| --- | --- |
| `models/` | Model components, tokenizer and preprocessing |
| `connectors/` | Connector architecture settings |
| `data/train/` | Training data fields, referenced under `data.from` |
| `data/eval/` | Benchmark input fields, merged at the evaluation root with `from` |
| `training/` | Training fields, referenced under `training.from` |
| `experiments/` | Training entries, with one or multiple stages |
| `eval/` | Benchmark entries |
| `deepspeed/` | Runtime JSON files consumed by DeepSpeed |

`training` contains Trainer fields directly. `training.tuning` is an independent
update-policy object; `training.tuning.peft` contains LoRA settings. Native mappings such as
`lr_scheduler_kwargs` retain their structure. `training.global_batch_size` derives
accumulation from worker count and micro batch; `launcher` controls devices,
distributed launch, and port. HF validates its options when worker arguments are
constructed; workflow checks cover project structure and parameters.

Algorithms live under `train/algorithms/`, and component update strategies under
`train/strategy/`. SFT reuses shared argument validation and batch preparation.
Future algorithms can replace algorithm-specific preparation or execution;
only SFT is currently implemented.

Use `--steps pretrain` or `--steps finetune` to execute one stage of an experiment.
Custom single-stage entries can omit `stages`, as in the first example. LoRA/QLoRA
presets only configure training; sequence length and attention backend stay under
`model`. The supplied adapter experiments inherit their model's values, without
requiring length 3072 or Flash Attention 2.

## Evaluation and external configs

```bash
python -m tinyllava.cli eval --config configs/eval/gqa.yaml \
  model_name_or_path=output/phi/finetune devices=[0,1] --dry-run
```

Evaluation parameters are flat. Reuse benchmark inputs with a root reference:

```yaml
# configs/eval/custom.yaml
# Benchmark inputs
from: ../data/eval/gqa.yaml
benchmark: gqa
image_folder: /path/to/gqa/images

# Model and generation
model_name_or_path: output/phi/finetune
temperature: 0.0
max_new_tokens: 128

# Worker execution
batch_size: 4
```

Paths and benchmark scoring belong to the evaluation workflow. `--steps generate` selects prediction
generation only. Outputs remain outside the input datasets.

An external `--config /path/to/experiment.yaml` follows the same path rules;
no registration or search-path configuration is needed. Composition does not
change the working directory.

`utils/config.py:compose_config` is the single composition entry point and uses
OmegaConf for merging, dotlist parsing, and interpolation. Workflows and algorithms
receive plain mappings. Workers read resolved YAML through `load_config`.
`--dry-run` prints resolved settings and commands without loading a model.
