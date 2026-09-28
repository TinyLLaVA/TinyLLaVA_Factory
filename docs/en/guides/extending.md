# Extending TinyLLaVA

Use the active interfaces in the [architecture](../architecture.md). Model
construction, tuning policy, data transformation, and scoring are separate
responsibilities.

| Extension | Entry point |
| --- | --- |
| Language model | Hugging Face text config and `language_model_name_or_path` |
| Vision tower | HF Auto registration, lazy `AutoVisionTowerModel`, and `AutoImageProcessor` |
| Connector | `tinyllava.model.connector.auto` config/model mappings |
| Dataset schema | `tinyllava.data.adapters` |
| Chat format | `model.chat_template_path` and Jinja generation blocks |
| Tuning policy | `tinyllava.train.strategy` |
| Evaluation task | `tinyllava.eval.tasks.auto` loader/evaluator mappings |

A connector must implement the current config/model contract and produce
features compatible with the language model's hidden size. Register its config
and model in the canonical auto mappings and test save/load round trips.

A custom chat template must render image placeholders consistently and mark
assistant output with `{% generation %}` blocks. Validate rendered prompts and
assistant masks before training to check which tokens contribute to the loss.

For new datasets, adapt the source schema to normalized messages and image
payloads. Keep dataset-specific path handling in the adapter.

Add API directives to the relevant [reference page](../reference/data.md) when
introducing public interfaces.

## Vision models and I-MoF

Native CLIP, SigLIP, DINOv2 and other compatible HF vision backbones load through
`AutoVisionTowerModel`, which delegates native models to the live HF Auto mapping.
TinyLLaVA extracts the vision sub-config from CLIP/SigLIP composite
configs. Custom models register their configuration and model with the public
HF Auto APIs. They must return spatial token hidden states and expose compatible
`hidden_size`, `patch_size` and image-token metadata. The saved preprocessing
configuration selects `AutoImageProcessor`; there is no second vision registry.

MoF comes from Tong et al., [Eyes Wide Shut? Exploring the Visual Shortcomings
of Multimodal LLMs](https://arxiv.org/abs/2401.06209) (CVPR 2024).
[TinyLLaVA Factory, Table 2](https://arxiv.org/html/2405.11788v1) cites this
method. Its upstream `mof_mlp.py` implements **I-MoF**: separate CLIP and DINOv2
MLPs followed by spatial token interleaving. The migrated implementation retains
that behavior, including the shared CLIP-preprocessed image for both branches.
This differs from A-MoF, which mixes features before projection.

- `MofVisionConfig` stores both branch configs; `MofVisionModel` has HF
  construction and save/load behavior, without downloads during construction.
- Branches must share patch size, channel count and depth. DINOv2 interpolates
  its positions to the CLIP input resolution. Branch hidden widths may differ.
- `MofConnector` projects each branch independently and produces tokens in
  `clip_0, dino_0, clip_1, dino_1, ...` order. Multiple selected hidden layers
  remain separate per branch before projection.
- `TinyLlavaProcessor` obtains placeholder counts from the connector config's
  `get_output_sequence_length` method. MoF maps N selected tokens to 2N;
  its name does not appear in processor selection or token expansion.
- All connectors receive `vision_hidden_size`, `text_hidden_size` and
  `vision_feature_layer`. Each connector validates its own input contract and
  delegates `get_output_sequence_length` to its config; the composite model has no MoF
  branch and does not choose a connector based on the vision model.

Prepare a reusable MoF vision checkpoint once:

```python
from transformers import AutoImageProcessor
from tinyllava.model.vision_tower.mof import MofVisionModel

clip_path = "openai/clip-vit-large-patch14-336"
vision = MofVisionModel.from_pretrained_components(
    clip_model_name_or_path=clip_path,
    dinov2_model_name_or_path="facebook/dinov2-large",
)
vision.save_pretrained("output/mof-vision")
AutoImageProcessor.from_pretrained(clip_path).save_pretrained("output/mof-vision")
```

MoF stores the two packed feature widths in its own connector config as
`vision_hidden_sizes` (CLIP first, DINOv2 second). Their sum must match
`vision_hidden_size`. Ordinary connectors receive only the input width,
not the vision model config. Change the two widths when using other branches.

Then select both the vision tower and its connector explicitly:

```bash
python -m tinyllava.run --config configs/train/pretrain.yaml \
  model.vision_model_name_or_path=output/mof-vision \
  model.connector_config=configs/connectors/mof.yaml model.vision_feature_select_strategy=default \
  name=qwen2_mof
```

The complete checkpoint includes both branches and projections. Use
`AutoVisionTowerModel` for deferred loading of a standalone MoF checkpoint.
Package and configuration imports leave model implementations unloaded; the
local lazy mapping resolves MoF only when it is selected. To use HF `AutoModel`
directly for MoF, explicitly import `MofVisionModel` to register its class.
Use `tinyllava.data.processor.AutoProcessor.from_pretrained` to load a saved
processor lazily. Direct HF `AutoProcessor` loading of a custom processor requires
an explicit `TinyLlavaProcessor` import to register its class.
The local tests cover small-model numerical behavior and save/load round trips;
they do not establish reproduction of the paper's benchmark scores.

## Custom image processors

`tinyllava.data.image_processor` is the lazy extension point used by training
and inference. HF-native image processors continue to use HF's own resolution.
For a project-specific processor, add a package such as
`tinyllava/data/image_processor/custom/` exporting `CustomImageProcessor`, then
add `("custom", "CustomImageProcessor")` to
`CUSTOM_IMAGE_PROCESSOR_MAPPING_NAMES` in `image_processor/auto/auto_mappings.py`.
Use a unique class name and save it as `image_processor_type` with
`save_pretrained`. Only the processor selected by this metadata is imported.
The local Auto class does not modify HF's global mappings or loaders.

## Connector implementations

Connector packages contain `configuration_<name>.py`, `modeling_<name>.py`,
and a lazy `__init__.py`. The old decorator registry and flat modules are removed.
Use `MLPConnectorConfig(depth=1)` for a linear projection. Identity validates
that the concatenated vision width equals the language-model width.

Resampler uses learned Perceiver queries. Q-Former uses the maintained HF
`Blip2QFormerModel`, initialized from config without downloading BERT. Both
produce `num_queries` tokens per image. All TinyLLaVA connectors use the
same `TinyLlavaProcessor`, which persists the connector config and calls its
sequence-length method. Only config classes are loaded during processor-only
loading; connector model weights and implementations are not needed.
These migrated architectures require newly trained connectors; old connector
weight layouts are not supported. Freezing is controlled by the tuning strategy.

The processor Auto mapping is keyed by the composite `TinyLlavaConfig`, as in
HF's model/processor relationship. Adding a connector length rule requires
implementing `get_output_sequence_length(input_length)` in its config, with no
new processor subclass or processor mapping entry. MLP/Identity preserve N,
MoF returns 2N, and query connectors return their configured fixed count.
Model-side splitting delegates to the same method, avoiding duplicate rules.
