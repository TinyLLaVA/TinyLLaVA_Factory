# 数据与 processor API

标注准备见[数据指南](../guides/data.md)，数据集选项见[参数速查](../guides/parameters.md#training-data)。

::: tinyllava.data.dataset
    options:
      members: [ProcessorSFTDataset, load_training_dataset, make_supervised_data_module]

::: tinyllava.data.adapters.llava_legacy.LlavaLegacyDatasetAdapter

::: tinyllava.data.collator
    options:
      members: [DataCollatorForMultimodalSFT, build_labels]

::: tinyllava.data.processor.processing_mixins.ComponentProcessorMixin

::: tinyllava.data.chat_template.loading.resolve_chat_template

::: tinyllava.data.processor.tinyllava.TinyLlavaProcessor

::: tinyllava.data.processor.auto.AutoProcessor
    options:
      members: [from_config, from_pretrained]

::: tinyllava.data.image_processor.auto.AutoImageProcessor
    options:
      members: [from_pretrained]
