# Changelog

## 2.0.0 — Unreleased

### Breaking changes

- Transformers now requires version 5.20.1 or newer, with the development and
  CI dependency versions recorded in `uv.lock`.
- Scheduler warmup uses `training.scheduler.warmup_steps`: values below 1
  specify a fraction of training steps, and values of 1 or more specify a step
  count. Replace `warmup_ratio: 0.03` with `warmup_steps: 0.03`.
- Length-based sampling uses `training.dataloader.train_sampling_strategy:
  group_by_length` instead of the `group_by_length` boolean. Removed Trainer
  fields are no longer accepted: `overwrite_output_dir`, `save_safetensors`,
  `logging_dir`, `include_tokens_per_second`, `half_precision_backend`,
  `use_mps_device`, and `fsdp_transformer_layer_cls_to_wrap`.
- The command entry point is `tinyllava` / `python -m tinyllava.cli`, with explicit
  `train`, `eval`, `single-turn`, and `chat` commands. `tinyllava.run` is removed.
- Configuration files use a unified `from` feature at any mapping node for model,
  connector, recipe, data, evaluation, and algorithm-specific sections. References
  can contain one path or a list; later sources, local fields, and CLI overrides
  take precedence in that order. Bare filenames at preset nodes select the matching
  directory under `configs/`; explicit relative paths are relative to their declaring
  YAML, or cwd for CLI overrides. File selections and
  value overrides both use `key=value`. Hydra defaults and its dependency are removed.
- Preset nodes support name shorthand: `model: phi` is equivalent to
  `model: {from: phi}`. References without a YAML extension get `.yaml` appended;
  explicit `.yaml` and `.yml` remain supported. Connector, training, and data nodes follow the same
  rule, including inside stages. The workflow selects `data/train/` or `data/eval/`.
  CLI shorthand supports nested overrides, e.g. `model=phi model.model_max_length=4096`.
  Use `./file.yaml` for a local file instead of a preset; missing presets do not fall back.
- Stage checkpoint dependencies use `from_stage`; `from` always denotes YAML
  composition. The resolved worker configuration contains no file references.
- Preset files match their destination node: `configs/training/` replaces
  `configs/recipes/` and exposes training fields directly for `training.from`.
  Training data lives under `configs/data/train/`; evaluation data under
  `configs/data/eval/` also exposes fields directly for `data.from`.
- Redundant `configs/train/` entries are removed. Use an experiment with `--steps`
  to run a selected stage, or provide a custom single-stage entry.
- LoRA/QLoRA settings no longer force length 3072 or Flash Attention 2. The supplied
  Phi adapter experiments inherit length 2048 and the model's default attention
  backend. Set these explicitly under `model` when needed.
- `experiment_name` names an experiment and its default output directory; it does
  not select a model. It replaces the ambiguous top-level `name` field.
- Training settings are grouped under `training`: `tuning`, `saving`, `logging`,
  `optimizer`, `scheduler`, `runtime`, `dataloader`, and `evaluation`. Core batch,
  learning-rate, epoch, and step settings remain directly under `training`.
  PEFT settings belong to `training.tuning.peft`; global batch size belongs to
  `training.global_batch_size`. Launcher settings only describe process launch.
- Training experiment YAML files explicitly select `algorithm: sft`. Custom configs
  that omit the algorithm currently default to SFT; no RL algorithm is implemented.
- The training worker consumes resolved YAML and dispatches directly to the
  algorithm. Parameter construction belongs to each algorithm. The generic
  `parse_train_arguments`, `read_train_config`, and `build_train_arguments`
  interfaces are removed. Trainer algorithms reuse common grouped argument construction in `hparams/training.py`.
- Algorithms now live under `train/algorithms/`; parameter update strategies remain
  under `train/strategy/`. Shared validation, field groups, and batch preparation
  belong to `hparams`, not to SFT.
- YAML composition lives in `utils.config`; training/evaluation workflows handle
  runtime paths and execution, and argument types live in `hparams`.
  `configuration.py` and `utils.arguments` are removed.

These changes describe the v2 development interface. Intermediate, unreleased
schemas have no compatibility shims or field-specific migration branches.
Invalid configuration is checked against the current schema.

See [the configuration reference](docs/en/reference/configuration.md) for the
current structure and [the migration guide](docs/en/migration.md) for v1 assets.
