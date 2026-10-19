# Changelog

## 2.0.0 — Unreleased

### Breaking changes

- Evaluation YAML is flat: checkpoint, data, generation, device, and output fields
  all live at the root. Construct `hparams.arguments.EvalArguments(**config)`
  directly. Benchmark input presets are merged with root `from` references.
  Worker CLI parsing lives in `eval/batch_generation.py`; `hparams/parser.py`
  and the per-section evaluation argument classes are removed. Direct worker
  configs must supply checkpoint and question-file paths.
- ScienceQA's option-letter instruction belongs to its benchmark loader.
  `data.single_pred_prompt` is removed, preserving the existing ScienceQA
  preset's prompt. Dataset loaders no longer receive the worker's `args` object.
- Project parameter types, including `EvalArguments` and `TrainingArgBundle`, live
  in `hparams.arguments`; only the HF `TrainingArguments` subclass lives in
  `hparams.hf`, imported when constructing training worker inputs.
  `hparams/tuning.py` is removed. Set `training.precision` for training
  dtype; derived HF `bf16` and `fp16` flags are no longer YAML options.
- Transformers supports `>=5.10.0,<6`. CI tests the minimum supported version
  and the latest release allowed by `pyproject.toml`; `uv.lock` records a
  reproducible development environment.
- Scheduler warmup uses `training.warmup_steps`: values below 1
  specify a fraction of training steps, and values of 1 or more specify a step
  count. Replace `warmup_ratio: 0.03` with `warmup_steps: 0.03`.
- Length-based sampling uses `training.train_sampling_strategy:
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
  rule, including inside stages. Training data names select `data/train/`; flat
  evaluation entries use explicit root references to `data/eval/`.
  CLI shorthand supports nested overrides, e.g. `model=phi model.model_max_length=4096`.
  Use `./file.yaml` for a local file instead of a preset; missing presets do not fall back.
- Stage checkpoint dependencies use `from_stage`; `from` always denotes YAML
  composition. The resolved worker configuration contains no file references.
- Preset files match their destination node: `configs/training/` replaces
  `configs/recipes/` and exposes training fields directly for `training.from`.
  Training data lives under `configs/data/train/`; evaluation data under
  `configs/data/eval/` exposes fields directly for root `from` references.
- Redundant `configs/train/` entries are removed. Use an experiment with `--steps`
  to run a selected stage, or provide a custom single-stage entry.
- LoRA/QLoRA settings no longer force length 3072 or Flash Attention 2. The supplied
  Phi adapter experiments inherit length 2048 and the model's default attention
  backend. Set these explicitly under `model` when needed.
- `experiment_name` names an experiment and its default output directory; it does
  not select a model. It replaces the ambiguous top-level `name` field.
- Trainer settings are direct fields under `training`, including `output_dir`,
  `logging_steps`, `warmup_steps`, and `precision`. `training.tuning` and
  `training.tuning.peft` remain independent configuration objects. Nested Trainer
  field groups and their whitelists are removed; HF validates its fields when
  worker arguments are built.
  `training.global_batch_size` derives gradient accumulation before worker launch.
- Multi-stage experiments select `algorithm` and `data` within each stage, with
  shared model, training, and launcher defaults at the root. Single-stage files
  place their algorithm and data at the root. Every shipped stage explicitly
  selects `algorithm: sft`; omitted algorithms default to SFT. No RL algorithm
  is implemented.
- `TrainingInputs` is renamed to `TrainingArgBundle`, a collection of model,
  data, Trainer, and tuning parameter objects.
- The training worker consumes resolved YAML and dispatches directly to the
  algorithm. Parameter construction belongs to each algorithm. The generic
  `parse_train_arguments`, `read_train_config`, and `build_train_arguments`
  interfaces are removed. Trainer algorithms reuse common argument construction in `hparams/training.py`.
- Algorithms now live under `train/algorithms/`; parameter update strategies remain
  under `train/strategy/`. Shared validation and batch preparation
  belong to `hparams`, not to SFT.
- YAML composition lives in `utils.config`; training/evaluation workflows handle
  runtime paths and execution, and argument types live in `hparams`.
  `configuration.py` and `utils.arguments` are removed.

See [Configuration](docs/en/guides/configuration.md) for the current v2 structure,
[Migration](docs/en/migration.md) for updating existing experiments, and the
[v2 roadmap](docs/en/roadmap.md) for planned work and its validation milestones.
