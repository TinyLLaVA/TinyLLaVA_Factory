# 模型与 checkpoint 加载

`TinyLlavaConfig` 包含文本、视觉和连接器子配置。使用 `from_pretrained` 加载已保存模型，使用 `from_pretrained_components` 组装初始组件。参见[训练](../guides/training.md)。

::: tinyllava.model.configuration_tinyllava.TinyLlavaConfig

::: tinyllava.model.modeling_tinyllava.TinyLlavaForConditionalGeneration
    options:
      members: [from_pretrained_components, forward, get_image_features, prepare_inputs_for_generation]

::: tinyllava.utils.model_loading
    options:
      members: [TinyLlavaModelBundle, ComponentPaths, resolve_component_paths, load_training_model, load_tinyllava_model_bundle, load_tinyllava_checkpoint_bundle]
