# v2 roadmap

The v2 roadmap covers release validation, DeepStack hooks, and native Qwen3-VL
integration. The table tracks implementation status and the next acceptance criteria.
Use the [migration guide](migration.md) to update an existing experiment.

| Workstream | Status | Next milestone |
| --- | --- | --- |
| Model, processor, and experiment interfaces | Implemented on the v2 development branch | Validate representative training, checkpoint reload, generation, and evaluation runs |
| Qwen3-VL-style DeepStack Hook plugin | Standalone prototype | Bind inputs to each call and preserve them through checkpoint recomputation |
| Native Qwen3-VL and dynamic-resolution inputs | Planned | Align packed patches, media grids, placeholders, positions, and batching end to end |

## Qwen3-VL / DeepStack Hook plugin { #deepstack-hooks }

[DeepStack](https://arxiv.org/abs/2406.04334) distributes visual information across
network depth. [Qwen3-VL](https://arxiv.org/abs/2511.21631) uses features from multiple
vision layers for this purpose. The proposed plugin applies that approach to
assembled HF language models: a decoder-layer `register_forward_hook` adds aligned
visual features at the visual positions in the layer output. The transformation
preserves `[batch, sequence, hidden]` shape and the host model's decoder loop,
attention, positional encoding, and cache implementation.

### Responsibilities

| Part | Responsibility |
| --- | --- |
| Vision encoder and connector | Produce the main and auxiliary visual features; align their dimensions and media/token order |
| Composite VLM | Select injection layers, map features to positions in the current call, and handle generation slicing and batch expansion |
| Layer adapter | Locate native decoder layers and update their hidden states while preserving Tensor or tuple output structure |
| Call context | Own each call's features and masks through forward and backward recomputation |
| Checkpoint configuration | Describe the extension and restore its hooks and trainable parameters during loading |

Auto mappings continue to select the language model class. The plugin adds
layer-level behavior to the selected instance. Sequence-length changes,
attention-topology changes, and multimodal positional encoding belong to
architecture-specific model adapters.

### Prototype evidence

The local prototype uses Torch 2.13.0, Transformers 5.13.1,
CPU/FP32, and tiny randomly initialized Llama, Qwen3, Gemma2, and OpenELM models.
Its visual inputs are differentiable tensors representing connector outputs.
The plugin is at the prototype stage; production training and generation integration
is covered by the milestones below.

| Check | Saved result for all four models |
| --- | --- |
| Zero injection | Logits match the original model exactly |
| Nonzero injection | Logits change and visual-feature gradients are nonzero |
| Reentrant and non-reentrant checkpointing | Logits, visual gradients, and model-parameter gradients match ordinary execution |
| Cached single-token decoding | Matches full-sequence execution |
| Parameter names and hook removal | Parameter keys stay unchanged; removal restores the original output |
| Removing hooks before backward | Reproduces `CheckpointError` during recomputation |

**Hooks and their call inputs must remain valid through backward recomputation.**
The prototype keeps a serial context alive until backward completes. The next
design must also preserve the correct context when several forward calls are pending.

### Integration milestones

1. **Call ownership.** Bind features and masks to each invocation. Test two forward
   calls followed by backward, gradient accumulation, and both checkpoint modes.
2. **Trainable feature path.** Connect a trainable connector to a frozen language
   model and compare connector gradients with ordinary execution.
3. **Generation and outputs.** Test `generate` with and without cache, beam
   expansion, padding, and multiple images. Determine injection from the visual
   positions present in the current call. Specify how capture hooks and
   `output_hidden_states` observe the injected values.
4. **Save and reload.** Restore extension configuration and trainable weights,
   preserve host parameter names, and check zero-injection equivalence after reload.
5. **Execution backends.** Validate mixed precision, distributed training, and
   compilation against the established training and generation checks.

The runtime integration should provide one shared layer transformation with small
adapters for layer location and output structure. Public configuration fields will
follow the validated call-lifecycle and serialization design.

## Native Qwen3-VL and dynamic resolution { #native-qwen3-vl }

Native Qwen3-VL integration provides an end-to-end baseline for component and
DeepStack experiments. Its processor, packed-patch layout, `image_grid_thw`,
modality token IDs, spatial mergers, positional encoding, and video timestamps
must stay aligned through preprocessing, batching, forward, and generation.

The vision feature handoff should identify the selected output, hidden dimension,
per-media lengths, and applied merging/projection. Main visual features and
DeepStack auxiliary features need explicit destinations in the composite model.
`ComponentProcessorMixin` already provides reusable configuration and length helpers;
each native processor controls where those helpers enter its own processing flow.

Acceptance starts with mixed-resolution image batches and text/image mixtures,
then expands to multiple images and video. Compare the integrated path with the
native model on fixed inputs, including cached generation and checkpoint reload.
