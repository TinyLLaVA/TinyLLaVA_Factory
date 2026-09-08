# Data and processor API

Adapters normalize source records. The processor owns multimodal encoding;
the collator pads tokens and concatenates image tensors.

::: tinyllava.data.dataset
    options:
      members: [ProcessorSFTDataset, load_training_dataset, make_supervised_data_module]

::: tinyllava.data.adapters.llava_legacy.LlavaLegacyDatasetAdapter

::: tinyllava.data.collator
    options:
      members: [DataCollatorForMultimodalSFT, build_labels]

::: tinyllava.data.processor.creation.create_tinyllava_processor

::: tinyllava.data.chat_template.loading.resolve_chat_template
