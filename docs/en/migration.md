# Migrating to v2

v2 keeps TinyLLaVA's vision encoder–connector–language model composition and gives
experiments shared model/processor interfaces, composable YAML, and explicit
training and evaluation workflows. This guide covers the current development
interface. The [v2 roadmap](roadmap.md) tracks DeepStack hooks and native Qwen3-VL integration.

Use [Configuration](guides/configuration.md) for YAML examples,
[Parameters](guides/parameters.md) for field meanings, and
[Models and checkpoints](guides/models.md) for component and checkpoint selection.

## Entry points and configuration

Run commands through `tinyllava` or `python -m tinyllava.cli`:

| Task | Command |
| --- | --- |
| Training | `python -m tinyllava.cli train --config configs/experiments/phi.yaml` |
| Evaluation | `python -m tinyllava.cli eval --config configs/eval/gqa.yaml model_name_or_path=output/phi/finetune` |
| Single-turn inference | `python -m tinyllava.cli single-turn --model-path output/phi/finetune --query "Describe the image." --image-file image.jpg` |
| Conversation | `python -m tinyllava.cli chat --model-path output/phi/finetune` |

For v1 shell recipes, transfer model, data, training, and launch settings into an
experiment YAML. For earlier v2 development configurations, apply these changes:

| Previous setting | Current setting |
| --- | --- |
| `python -m tinyllava.run` | `python -m tinyllava.cli` with a task subcommand |
| Model selection through `model.name` | `model: phi` or `model: {from: phi, ...}` |
| YAML references using `from_config`, `from_yaml`, or Hydra `defaults` | `from` at the destination node; supports a path or ordered list of paths |
| Experiment YAML `model.connector_config` | `model.connector`, with the same `from` and override rules |
| `configs/recipes/` | `configs/training/`; preset fields go directly under `training` when referenced |
| `configs/train/` entries | An experiment plus `--steps`, or a single-stage YAML |
| Root `name` | `experiment_name` |
| Root `algorithm` and `data` in a multi-stage experiment | `stages.<name>.algorithm` and `stages.<name>.data` |
| `data.data_path` | `data.dataset_name_or_path` |
| Stage dependency written as `from` | `from_stage`; `from` composes YAML |
| `training.saving.*`, `training.logging.*`, and other Trainer field groups | Direct `training.*` fields, retaining `training.tuning` and its `peft` mapping |
| Evaluation `model.*`, `data.*`, `generation.*`, `runtime.*`, `output.*` | The same field names at the evaluation YAML root |
| Evaluation `data.from` | Root `from` with a path to `configs/data/eval/` |
| `single_pred_prompt` | Remove the option; the ScienceQA loader supplies its option-letter instruction |

These YAML changes apply to experiment configuration. Saved HF model configuration
continues to use `text_config`, `vision_config`, and `connector_config`.
The [v2 changelog](https://github.com/TinyLLaVA/TinyLLaVA_Factory/blob/develop/CHANGELOG.md)
records the breaking changes; loaders validate configuration against the current schema.

### Training example

```yaml
# configs/experiments/custom.yaml
model: phi
experiment_name: phi_trial

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
      save_steps: 500
      logging_steps: 10
      tuning:
        peft:
          r: 64
          lora_alpha: 128
```

Model, training, and launcher defaults can be shared at the root. Each stage
selects its algorithm and dataset. Later stages load the preceding stage's
checkpoint; `from_stage` selects an earlier stage and `checkpoint` selects an
external source. Run prerequisite stages before selecting a later stage on its own.

In a single-stage entry, place `algorithm`, `data`, `model`, and `training` at the
root. Use `stage` to name its output stage. Training data presets live under
`configs/data/train/`; model and connector presets live under `configs/models/`
and `configs/connectors/`.

### Evaluation example

```yaml
# configs/eval/custom.yaml
from: ../data/eval/gqa.yaml
benchmark: gqa
model_name_or_path: output/phi_trial/finetune
batch_size: 4
temperature: 0.0
output_dir: output/phi_trial/finetune/eval/gqa
```

The evaluation workflow resolves paths, assigns shards when requested, and writes
predictions to `answers_file`, normally `${output_dir}/answers.jsonl`.
A direct `tinyllava.eval.batch_generation` worker takes resolved root fields,
including `model_name_or_path` and `question_file`.

## Trainer and tuning settings

The supported Transformers range is `>=5.10.0,<6`. Update these Trainer options
alongside the YAML structure:

| Previous setting | Current setting |
| --- | --- |
| `warmup_ratio: 0.03` | `warmup_steps: 0.03`; values below `1` are fractions, values at least `1` are step counts |
| `group_by_length: true` | `train_sampling_strategy: group_by_length` |
| Independent `bf16` / `fp16` flags | `precision: auto`, `bf16`, `fp16`, or `fp32` |
| `training.runtime.precision` | `training.precision` |
| `training.saving.resume_from_checkpoint` | `training.resume_from_checkpoint` |

Remove obsolete HF fields: `overwrite_output_dir`, `save_safetensors`,
`logging_dir`, `include_tokens_per_second`, `half_precision_backend`,
`use_mps_device`, and `fsdp_transformer_layer_cls_to_wrap`.

`training.global_batch_size` derives accumulation from worker count and per-device
batch size. `training.tuning` selects freezing and adapter policies. Set model
length and attention backend under `model`; the supplied Phi LoRA/QLoRA experiments
inherit the model preset's length of 2048 and default attention backend.

## Python extensions

| Responsibility | Current interface |
| --- | --- |
| YAML composition | `tinyllava.utils.config.compose_config` |
| Stage dependencies and evaluation paths | `tinyllava.train.workflow` and `tinyllava.eval.workflow` |
| Project parameter objects | `tinyllava.hparams.arguments`, including flat `EvalArguments` |
| HF Trainer extension | `tinyllava.hparams.hf.TrainingArguments` |
| Trainer input bundle | `TrainingArgBundle`, replacing `TrainingInputs` |
| Training argument construction | `tinyllava.hparams.training.build_training_arguments` |
| Training algorithms | Modules under `tinyllava.train.algorithms` |
| Component update policies | `tinyllava.train.strategy` |
| Processor selection and loading | `tinyllava.data.processor.AutoProcessor` |
| Reusable processor configuration and length rules | `ComponentProcessorMixin` |
| LLaVA processing and model-based processor creation | `TinyLlavaProcessor` |

Update imports from the removed `configuration.py`, `utils.arguments`,
`hparams.parser`, and `hparams.tuning` modules. Evaluation constructs
`EvalArguments(**resolved_worker_config)`; the old per-section evaluation classes
are consolidated into that object. Dataset loaders receive their task inputs;
ScienceQA prompt formatting belongs to its loader.

Replace imports from `data.processor.processing_base` and `data.processor.creation`
with the processor interfaces above. `AutoProcessor.from_config(config, **kwargs)`
selects a class by model config and passes `kwargs` to its constructor.
Use `TinyLlavaProcessor.from_model` to create the LLaVA processor from model components.
Connector implementations now live in packages with config/model Auto mappings;
update extensions that used the old decorator registry or flat modules.

Register component config/model pairs through the corresponding Auto interfaces.
Project model types use namespaced identifiers such as `mlp__tlf_connector`;
implementations load when selected. New components must satisfy hidden-size,
feature-layout, processor, and generation contracts. The
[extension guide](guides/extending.md) covers registration and length methods.

## Checkpoints and input behavior

Save and reload the model together with its tokenizer, image processor, and chat
template. An explicit template override takes precedence over the saved template.
Phi and OpenELM presets select the bundled LLaVA template; the Qwen2 base legacy
experiment selects its historical templates. `pretrain_legacy` is an image-to-answer
format that omits the user's question, intended for that pretraining recipe.

`vision_feature_select_strategy` may be omitted. New assemblies infer it from the
vision configuration; checkpoint loads retain the saved value. SigLIP uses `full`,
while CLIP, DINO, and MoF use `default` to drop the leading CLS token. An explicit
value overrides the selection.

The loader remaps known legacy language-model, vision-encoder, and MLP weight keys.
Provide the matching configuration and preprocessing assets when migrating those
weights. Rebuilt query-based connectors require weights matching their current
architecture. Initialize a new stage with `checkpoint`; use
`training.resume_from_checkpoint` to restore a complete checkpoint's training state.

Keep original v1 experiments in a separate environment using `legacy/v1`; the
historical `doc` branch contains v1 Sphinx documentation.

## Migration checks

1. **Record the baseline.** Save the original environment, component/data versions,
   templates, training and generation settings, and outputs for a fixed sample set.
2. **Preview configuration.** Run the new entry with `--dry-run` and inspect the
   resolved presets, stage inputs, output paths, and effective batch size.
3. **Check processed samples.** Inspect questions, image placeholders, assistant
   masks, truncation, visual feature selection, and connector output lengths.
4. **Reload the checkpoint.** Save and reload the model and processor, investigate
   missing or unexpected weight keys, and compare fixed-input outputs at matching precision.
5. **Run a short training job.** Check trainable parameters, finite gradients, loss,
   checkpoint resume, and stage handoff.
6. **Evaluate against the baseline.** Match task definitions and scoring. Record
   changes in data, templates, preprocessing, generation budget, and resource use.

Document reproduction results with matched experimental settings. Measure startup,
training, and generation performance separately when reporting efficiency changes.
