# 数据与 processor API

adapter 归一化记录，processor 编码，collator 完成 padding 与监督标签。

::: tinyllava.data.dataset
    options:
      members: [ProcessorSFTDataset, load_training_dataset, make_supervised_data_module]

::: tinyllava.data.adapters.llava_legacy.LlavaLegacyDatasetAdapter

::: tinyllava.data.collator
    options:
      members: [DataCollatorForMultimodalSFT, build_labels]

::: tinyllava.data.processor.processing_base.BaseProcessor

::: tinyllava.data.chat_template.loading.resolve_chat_template

::: tinyllava.data.processor.tinyllava.TinyLlavaProcessor

::: tinyllava.data.processor.auto.AutoProcessor
    options:
      members: [from_config, from_pretrained]

::: tinyllava.data.image_processor.auto.AutoImageProcessor
    options:
      members: [from_pretrained]
