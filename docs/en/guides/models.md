# Models and checkpoints

A TinyLLaVA model combines a language model, a vision encoder, and a connector.
Select a component preset for initial training, or load a trained checkpoint
to continue training, evaluate, or chat.

## Component presets

Set `model: phi` in an experiment, or override it with `model=qwen2_instruct`.
The shipped presets select these components:

| Preset | Language model | Vision encoder |
| --- | --- | --- |
| `phi` | `microsoft/phi-2` | SigLIP SO400M, 384 px |
| `qwen2_base` | `Qwen/Qwen2-0.5B` | SigLIP SO400M, 384 px |
| `qwen2_instruct` | `Qwen/Qwen2-0.5B-Instruct` | SigLIP SO400M, 384 px |
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | SigLIP SO400M, 384 px |
| `stablelm` | `stabilityai/stablelm-2-zephyr-1_6b` | CLIP ViT-L/14, 336 px |
| `gemma` | `google/gemma-2b-it` | SigLIP SO400M, 384 px |
| `openelm` | `apple/OpenELM-270M-Instruct` | SigLIP SO400M, 384 px |

All these presets use `mlp2x_gelu`. OpenELM explicitly uses the Llama-2 tokenizer;
other presets use the language model's tokenizer unless overridden.

Override a preset locally to change components or preprocessing:

```yaml
model:
  from: phi
  model_max_length: 4096
  attn_implementation: sdpa
  connector:
    from: mlp2x_gelu
    depth: 3
```

`model_max_length` controls training truncation. `attn_implementation` selects a
backend supported by the language model and runtime. See the
[model parameters](parameters.md#model) for tokenizer and template overrides.

## Connectors and vision features

| Connector preset | Purpose | Main options |
| --- | --- | --- |
| `mlp2x_gelu` | Project visual features into language-model embeddings | `depth`, `act`, `bias` |
| `identity` | Pass through features with compatible dimensions | — |
| `qformer` | Extract a fixed set of visual query tokens | `num_queries` |
| `resampler` | Resample visual features into a fixed token count | `num_queries` |
| `mof` | Combine the two MoF vision branches | `vision_hidden_sizes`, `clip_config`, `dinov2_config` |

`model.vision_feature_layer` selects the hidden layer or layers. Without an explicit
`vision_feature_select_strategy`, SigLIP retains all patches; CLIP, DINO, and MoF
drop the leading CLS token. Loading a checkpoint preserves its saved strategy.
Use `full` to retain every token or `default` to drop the first token explicitly.
Custom vision and connector combinations are covered in the [extension guide](extending.md).

## Load a checkpoint

| Task | Checkpoint setting |
| --- | --- |
| Initialize a training stage | `stages.finetune.checkpoint=/path/to/checkpoint` |
| Evaluate | `model_name_or_path=/path/to/checkpoint` |
| Single-turn inference or chat | `--model-path /path/to/checkpoint` |

Later training stages load the preceding stage's output automatically. Use
`from_stage` to choose an earlier stage or `checkpoint: null` to assemble components
again. A checkpoint supplies its saved component configurations and weights.
`training.resume_from_checkpoint` additionally restores training state; see
[training and resuming](training.md#distributed-training-and-resuming).

Use the [inference guide](inference.md) for Python loading and generation examples,
or the [model API](../reference/model.md) for class and method signatures.
