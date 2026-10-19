# 模型与 checkpoint 加载

组件与 checkpoint 的选择见[模型指南](../guides/models.md)，加载与生成示例见[推理指南](../guides/inference.md)。

::: tinyllava.model.configuration_tinyllava.TinyLlavaConfig

::: tinyllava.model.modeling_tinyllava.TinyLlavaForConditionalGeneration
    options:
      members: [from_pretrained_components, forward, get_image_features, prepare_inputs_for_generation]

::: tinyllava.utils.model_loading
    options:
      members: [TinyLlavaModelBundle, ComponentPaths, resolve_component_paths, load_training_model, load_tinyllava_model_bundle, load_tinyllava_checkpoint_bundle]

::: tinyllava.model.vision_tower.mof.MofVisionConfig

::: tinyllava.model.vision_tower.mof.MofVisionModel
    options:
      members: [from_pretrained_components, forward]

::: tinyllava.model.connector.mof.MofConnector

::: tinyllava.model.llm.openelm.OpenELMConfig

::: tinyllava.model.llm.openelm.OpenELMForCausalLM
