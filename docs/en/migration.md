# Migrating from v1 to v2

## Why migrate?

The current `main` branch was originally developed primarily as a lightweight
research reproduction codebase for the TinyLLaVA paper. The scope of v1 was to
reproduce the paper's training and evaluation workflows. Its implementation
prioritized those experiments: training scripts fixed GPU assignments, precision,
batch settings, and output paths, while experiment-specific branches and unused
code accumulated alongside the active implementation.

The original input pipeline also took shape before today's mature conventions
for multimodal processors and
[Jinja chat templates](https://huggingface.co/docs/transformers/chat_templating_writing).
It maintained local formatter and template classes, with tokenization and image
preprocessing split across components. Modern Transformers processors own the
multimodal chat template and image-token expansion, so model-provided formatting
can travel with the checkpoint and be reused across applications.

Maintaining the research-era interfaces now requires repeated project-specific
adaptation as community models, checkpoint formats, and training APIs evolve.
The purpose of v2 is to move that foundation onto Hugging Face model and processor
interfaces, making community integrations easier to maintain. Training,
evaluation, and serving share loading and generation paths, and experiment
settings are recorded in YAML.

The migration has three goals:

- **Consistent checkpoints:** represent language, vision, and connector components
  in a composite configuration with standard loading and saving interfaces.
- **Consistent inputs:** let one processor manage tokenization, image processing,
  chat templates, and image-token alignment.
- **Focused extension points:** separate model loading, tuning policy, data
  adaptation, and evaluation so each can be extended independently.

## Migration path in `develop`

The implementation follows four connected changes:

1. **Model foundation.** [HF Auto alignment](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/955f5d9)
   replaces factory-only loading with config/model mappings, nested component
   configs, and lazy imports.
2. **Templates and inputs.** [Jinja template integration](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/bf7aa08)
   replaces the local template classes; [processor-based SFT](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/ed74649)
   moves encoding and assistant masks onto the shared processor path.
3. **Training configuration and policy.** [Structured YAML](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/c093160)
   centralizes experiment settings, while [strategy plugins](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/f2a42bf)
   separate tuning policy from loading and saving.
4. **Evaluation and serving.** [Shared batch generation](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/d8b57cc)
   and the [Web demo migration](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/a599089)
   bring benchmark and interactive use onto the same checkpoint and generation interfaces.

## Interface changes

| v1 | v2 |
| --- | --- |
| Local language, vision, and connector factories | HF-style config/model mappings and composite model |
| Loading and saving owned by training recipes | Model-loading utilities, tuning strategies, and checkpoint utilities |
| `conv_version` template registry | Processor chat template selected with `model.chat_template_path` |
| Separate text and image preprocessing | Processor encoding and assistant-token masks |
| Command-specific training options | YAML configuration with dotlist overrides |
| Task-specific generation entry points | Shared batch generation with task loaders and evaluators |
| `tinyllava.serve.cli` | `python -m tinyllava.eval.single_turn` |

## Update an experiment

1. Move model, data, and training settings into the corresponding YAML sections.
   Start from a configuration in `configs/train/`.
2. Set the component model paths for pretraining. For fine-tuning, set
   `model.pretrained_model_name_or_path` to the pretrained checkpoint.
3. Select the chat template and verify rendered prompts, image tokens, and
   assistant loss masks on a small batch.
4. Recalculate gradient accumulation for the GPU count to preserve the global batch.
5. Run generation and benchmark scoring with the matching evaluation configuration.

The Qwen2 base legacy recipe provides the paper's prompt formats. See
[Training](guides/training.md) and [Evaluation](guides/evaluation.md).

## Compatibility and limits

**Checkpoints.** The loader registers weight-key mappings for known legacy
language-model, vision-tower, and connector layouts. These mappings rename
weights; they do not supply missing configuration or processor assets.
A migrated checkpoint needs compatible text, vision, and connector configs,
weights, tokenizer, image processor, and chat template. Verify a load/save
round trip and compare outputs on fixed inputs. Use `legacy/v1` to run an
original checkpoint with its original runtime.

**Training state.** Loading model weights starts a new training stage.
Optimizer, scheduler, and distributed state from an older runtime require
separate compatibility checks before resuming. The supported resume path uses
complete checkpoints from the current training stack.

**Experiment results.** This release changes software interfaces, not the
published benchmark claims. Prompt rendering, image preprocessing, supervision
masks, effective batch, and generation settings determine whether an experiment
is comparable. Re-evaluate a migrated checkpoint using the same dataset split
and scoring method before comparing it with the paper's results.

**Extensions.** The removed factory and recipe APIs require adaptation to the
new interfaces. Available components and registration points are described in
the [extension guide](guides/extending.md).

## Documentation

The current documentation is built alongside the implementation. The historical
`doc` branch contains the standalone Sphinx documentation for v1 and serves as
an archive. Use the guides on this site for v2 configurations and commands.
