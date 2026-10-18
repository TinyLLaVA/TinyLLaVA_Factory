# Configuration API

Training and evaluation share `tinyllava.hparams.arguments`. Custom training
arguments extend HF TrainingArguments. The reference below lists project-defined fields.

Training and evaluation workflows call `utils.config.compose_config` to compose experiment YAML. The training worker in `train/train.py` reads resolved YAML, applies launcher overrides, and selects the algorithm. Evaluation arguments are parsed by `hparams.parser`.

::: tinyllava.hparams.parser
    options:
      members: [parse_eval_arguments, build_eval_arguments]

::: tinyllava.hparams.arguments
    options:
      members: [ModelArguments, DataArguments, TrainingArguments, EvalModelArguments, EvalDataArguments, EvalGenerationArguments, EvalRuntimeArguments, EvalOutputArguments, EvalArguments]

::: tinyllava.hparams.tuning
    options:
      members: [TuningArguments]

::: tinyllava.hparams.training
    options:
      members: [TrainingInputs, build_training_arguments, prepare_training_config]

::: tinyllava.utils.config
    options:
      members: [compose_config, load_config, merge_mappings, resolve_interpolations]

Configuration responsibilities:

| Field | Responsibility |
| --- | --- |
| `algorithm` | Training algorithm; currently only `sft`, also the default |
| `model` | Model components, tokenizer, templates, and vision features |
| `data` | Dataset inputs and processing |
| `training.tuning` | Update policy, frozen components, and LoRA parameters |
| `training` | Core steps, epochs, learning rate, and batch settings |
| `training.saving` | Output directory, checkpoints, and resuming |
| `training.logging` | Logging, run name, and reporting backends |
| `training.optimizer` / `training.scheduler` | Optimizer details and learning-rate scheduling |
| `training.runtime` | Precision, DeepSpeed, gradient checkpointing, and distributed runtime |
| `training.dataloader` / `training.evaluation` | Loading, sampling, and training-time evaluation |
| `launcher` | Processes, devices, and port; excluded from algorithm arguments |

`training.scheduler.warmup_steps` accepts a fraction below 1 (e.g. `0.03` for
3% of training steps) or a step count (e.g. `100`). Select length-based sampling
with `training.dataloader.train_sampling_strategy: group_by_length`.

TinyLLaVA's `from` feature composes YAML files at any mapping node. A single path
or a list of paths is accepted. Sources merge in order, followed by local fields,
then CLI overrides. Mappings merge recursively; lists and scalars are replaced.
Bare filenames use a preset directory: `model` → `configs/models/`, `connector` →
`configs/connectors/`, `training` → `configs/training/`, and `data` →
`configs/data/train/` or `configs/data/eval/` according to the workflow. This also
applies within stages and to external entry files. Explicit paths (`./`, `../`,
absolute paths or `~`) retain their original path rules: relative to the declaring
YAML, or cwd for CLI overrides. No local fallback is attempted for preset names.
At other nodes, `from` remains relative to its declaring file.

At preset nodes, `model: phi` is equivalent to `model: {from: phi}`.
Omitting the extension appends `.yaml`, including for explicit paths such as
`./custom`; explicit `.yaml` and `.yml` still work. This also supports `connector`, `training`, and
`data`; ordinary file-valued fields remain strings. CLI shorthand and deep
overrides compose together: `model=phi model.model_max_length=4096`.

Select files with `model.from=phi` or
`model.connector.from=mof`; use `key=value` to add or
replace any field. A replacement reference keeps explicit sibling settings while
removing the old file's defaults. `from` takes literal paths and is consumed by
composition. Value interpolation (`${dataset_dir}`, `${oc.env:NAME,default}`)
resolves after workflow runtime defaults are assigned. Cycles, missing files,
and non-mapping references are errors. Files can be reused in multiple nodes.

`experiment_name` or the entry filename names the experiment independently of the
model. Stage checkpoint dependencies use `from_stage`. Training files contain
training fields at their root and are referenced under `training.from`. Training
and evaluation data presets live in `configs/data/train/` and `configs/data/eval/`;
both expose data fields directly for `data.from`. For example,
`training: {from: [pretrain, ./hardware]}` combines training settings.

```yaml
# configs/experiments/custom.yaml
algorithm: sft
model:
  from: phi
  model_max_length: 4096
  connector:
    from: mlp2x_gelu
    depth: 3
    bias: false
training:
  from: lora_finetune
  per_device_train_batch_size: 4
  global_batch_size: 32
  learning_rate: 0.0002
  saving:
    output_dir: output/phi/sft
    save_steps: 500
  logging:
    logging_steps: 10
    report_to: tensorboard
  tuning:
    strategy: lora
    tune_type_llm: lora
    tune_type_vision_tower: frozen
    tune_type_connector: full
    peft:
      r: 128
      lora_alpha: 256
launcher:
  distributed: true
  devices: 0,1
```

The shared Trainer adapter resolves this batch configuration to `gradient_accumulation_steps: 4`.
An explicit `global_batch_size` overrides existing accumulation settings and is
removed from the resolved worker configuration.

`utils/config.py:compose_config` is the only file-composition entry point.
Workers read already composed YAML with `load_config`; they do not expand references again.
The training workflow handles stage inheritance, checkpoint dependencies, and launch.
Stage values win over shared values, including values contributed by stage references.

`hparams/training.py` owns shared grouping, validation, batch derivation, and HF
argument adaptation. `hparams/tuning.py` defines update-policy fields.
`train/algorithms/sft.py` reuses these functions and implements supervised training;
`train/algorithms/sft_trainer.py` contains its sampling and Trainer extensions.
Component update strategies remain in `train/strategy/`.
New algorithms may add sections or replace preparation/execution without copying
common field tables. Only SFT is implemented.

The common adapter returns `TrainingInputs(model, data, training, tuning)`;
RL algorithms can use their own input objects.

Evaluation uses `model.model_name_or_path` for checkpoints and `model.model_id`
for prediction identifiers. A reference such as `model: ./eval-model.yaml` must
load a mapping of evaluation-model fields; checkpoint paths belong in
`model.model_name_or_path`.

Every entry under `configs/experiments/` explicitly selects
`algorithm: sft`. Custom configs that omit it still default to SFT.
`experiment_name` controls default output directories and log names, independently
of model selection. Use `--steps` to run one stage, or provide a single-stage entry
without `stages`. Training presets never change model length or attention settings.

Groups remain nested in YAML and dry-run output. Only the shared Trainer adapter flattens
them when constructing HF `TrainingArguments`; tuning becomes a separate
`TuningArguments` object consumed by update strategies.
