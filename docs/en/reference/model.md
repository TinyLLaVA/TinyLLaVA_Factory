# Model and checkpoint loading

`TinyLlavaConfig` contains text, vision, and connector sub-configs. Load saved
models with `from_pretrained`; assemble new components with
`from_pretrained_components`. See [Training](../guides/training.md).

::: tinyllava.model.configuration_tinyllava.TinyLlavaConfig

::: tinyllava.model.modeling_tinyllava.TinyLlavaForConditionalGeneration
    options:
      members: [from_pretrained_components, forward, get_image_features, prepare_inputs_for_generation]

::: tinyllava.utils.model_loading
    options:
      members: [TinyLlavaModelBundle, ComponentPaths, resolve_component_paths, load_training_model, load_tinyllava_model_bundle, load_tinyllava_checkpoint_bundle]
