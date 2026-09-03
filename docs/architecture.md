# TinyLLaVA Factory Architecture

This document describes the current `develop` branch architecture.

## Current `develop` Architecture

```mermaid
flowchart TB
    user[User CLI]

    subgraph cfg[Configuration Layer]
        train_yaml[Train YAML<br/>configs/train]
        eval_yaml[Eval YAML<br/>configs/eval]
        zero_cfg[DeepSpeed ZeRO config]
        parsers[Config parsers<br/>train + eval]
    end

    subgraph train[Training Runtime]
        train_entry[tinyllava.train.train]
        strategy[train.strategy<br/>tuning + PEFT policy]
        hf_trainer[TinyLlavaTrainer<br/>HF Trainer subclass]
    end

    subgraph data[Data Pipeline]
        train_data[tinyllava.data]
        data_adapters[data adapters<br/>JSON streaming + normalization]
        collator
    end

    subgraph proc_align[Processor Alignment]
        hf_processor[HF processor<br/>tokenizer + image_processor + chat_template]
        image_token[image token + patch/token expansion]
    end

    subgraph model[Model Core]
        model_load[utils.model_loading]
        tiny_model[TinyLlavaForConditionalGeneration]
        components[LLM + vision tower + connector]
    end

    subgraph eval[Evaluation Runtime]
        batch[tinyllava.eval.batch_generation]
        eval_tasks[tinyllava.eval.tasks<br/>loaders + evaluators]
        answers[answers / submissions]
    end

    subgraph serving[Serving / Interactive]
        serve_runtime[tinyllava.serve]
    end

    user --> train_yaml --> parsers --> train_entry
    user --> eval_yaml --> parsers --> batch
    zero_cfg --> parsers

    train_entry --> train_data
    train_data --> data_adapters --> hf_processor --> collator
    model_load --> hf_processor --> image_token --> tiny_model
    train_entry --> strategy --> tiny_model
    train_entry --> model_load --> tiny_model --> components
    train_entry --> hf_trainer
    collator --> hf_trainer --> tiny_model

    batch --> model_load
    batch --> eval_tasks
    batch --> hf_processor
    batch --> tiny_model
    eval_tasks --> answers

    serve_runtime --> model_load
    serve_runtime --> hf_processor
    serve_runtime --> tiny_model
```

## Core Execution Paths

```mermaid
sequenceDiagram
    autonumber

    participant CLI as uv/deepspeed/python CLI
    participant Config as YAML config parser
    participant Loader as utils.model_loading
    participant Processor as HF processor
    participant Model as TinyLlavaForConditionalGeneration
    participant Data as data.dataset + adapters
    participant Strategy as train.strategy
    participant Trainer as TinyLlavaTrainer

    CLI->>Config: --config configs/train/models/*.yaml + overrides
    Config->>Loader: ModelArguments / DataArguments / TrainingArguments
    Loader->>Model: from_pretrained(checkpoint) or from_pretrained_components(...)
    Loader->>Processor: create tokenizer + image_processor + chat_template processor
    Processor->>Model: align image token id, embedding size, patch/token expansion
    Config->>Data: data_path / image_folder / dataset_adapter
    Data->>Processor: stream/adapt JSON, normalize messages, attach image payloads
    Config->>Strategy: training_strategy + tune policies + PEFT config
    Strategy->>Model: freeze/full/partial/LoRA preparation
    Processor->>Trainer: ProcessorSFTDataset + DataCollatorForMultimodalSFT
    Model->>Trainer: HF-compatible PreTrainedModel
    Trainer->>Trainer: train(resume_from_checkpoint=auto)
    Strategy->>Trainer: save_state + save_model
```

```mermaid
sequenceDiagram
    autonumber

    participant CLI as eval python CLI
    participant Config as eval YAML parser
    participant Batch as eval.batch_generation
    participant Tasks as eval.tasks loaders
    participant Loader as utils.model_loading
    participant Processor as HF processor
    participant Gen as eval.generation
    participant Out as answer/submission files

    CLI->>Config: --config configs/eval/*.yaml + dotlist overrides
    Config->>Batch: EvalArguments
    Batch->>Loader: load_tinyllava_checkpoint_bundle(model_name_or_path)
    Loader->>Processor: load checkpoint processor contract
    Batch->>Tasks: AutoDatasetLoader.from_name(adapter)
    Tasks->>Batch: normalized GenerationExample
    Batch->>Gen: messages + images + processor + generation options
    Gen->>Batch: decoded response
    Batch->>Tasks: process_response(...)
    Tasks->>Out: JSONL answers, then benchmark-specific conversion/scoring
```

## Model Core

```mermaid
classDiagram
    class TinyLlavaConfig {
        text_config
        vision_config
        connector_config
        vision_feature_layer
        vision_feature_select_strategy
    }

    class TinyLlavaForConditionalGeneration {
        +from_pretrained()
        +from_pretrained_components()
        +generate()
        +forward()
    }

    class TinyLlavaModel {
        language_model
        vision_tower
        multi_modal_projector
        +get_image_features()
        +forward()
    }

    class AutoLanguageModel {
        +from_config()
        +from_pretrained()
    }

    class AutoVisionTowerModel {
        +from_config()
        +from_pretrained()
    }

    class AutoConnectorModel {
        +from_config()
    }

    class MLPConnector {
        +forward()
    }

    class MOFVisionTower {
        +from_pretrained()
        +forward()
    }

    TinyLlavaForConditionalGeneration --> TinyLlavaConfig
    TinyLlavaForConditionalGeneration *-- TinyLlavaModel
    TinyLlavaModel *-- AutoLanguageModel
    TinyLlavaModel *-- AutoVisionTowerModel
    TinyLlavaModel *-- AutoConnectorModel
    AutoConnectorModel ..> MLPConnector
    AutoVisionTowerModel ..> MOFVisionTower
```

The important direction in `develop` is that TinyLLaVA now behaves like a
composite Hugging Face model. A complete TinyLLaVA checkpoint should load through
`TinyLlavaForConditionalGeneration.from_pretrained(...)`; fresh pretraining still
assembles language, vision, and connector components through
`from_pretrained_components(...)`.

## Processor Alignment

The processor boundary follows the direction introduced by `955f5d9` and
`ed74649`: TinyLLaVA should expose one Hugging Face processor contract instead of
asking callers to manually compose tokenizer, image processor, chat template,
image-token expansion, and assistant-label repair. Training data is normalized
into multimodal chat messages, image payloads are attached to those messages, and
the processor owns tokenization plus image preprocessing before the collator and
Trainer see the batch.

## `develop` vs `main`

```mermaid
flowchart LR
    subgraph main[main: old architecture]
        main_args[CLI arguments]
        main_factories[local factories]
        main_recipe[training_recipe]
        main_data[eager JSON dataset]
        main_preprocess[split text/image preprocess]
        main_eval[task-specific eval entrypoints]

        main_args --> main_factories
        main_args --> main_recipe
        main_data --> main_preprocess
        main_data --> main_recipe
        main_eval --> main_factories
    end

    subgraph develop[develop: current architecture]
        dev_yaml[structured YAML]
        dev_loader[model_loading]
        dev_processor[HF processor contract]
        dev_hf[HF-style TinyLLaVA model]
        dev_strategy[train.strategy]
        dev_data[data adapters]
        dev_eval[eval.tasks]

        dev_yaml --> dev_loader --> dev_processor --> dev_hf
        dev_yaml --> dev_strategy --> dev_hf
        dev_yaml --> dev_data --> dev_processor
        dev_yaml --> dev_eval --> dev_hf
    end

    main_args -.replaced by.-> dev_yaml
    main_factories -.replaced by.-> dev_loader
    main_preprocess -.aligned as.-> dev_processor
    main_recipe -.split into.-> dev_strategy
    main_data -.reworked as.-> dev_data
    main_eval -.consolidated as.-> dev_eval
```

| Area | `main` | `develop` |
| --- | --- | --- |
| Configuration | CLI flags parsed by `HfArgumentParser`; defaults are scattered across command invocations. | Structured YAML under `configs/train` and `configs/eval`, with OmegaConf dotlist overrides. |
| Model assembly | Instantiate empty `TinyLlavaForConditionalGeneration`, then call `load_llm`, `load_vision_tower`, `load_connector`. | Prefer HF loading: complete checkpoints through `from_pretrained`; fresh assembly through `from_pretrained_components`. |
| Component resolution | Project-local factories: `LLMFactory`, `VisionTowerFactory`, `ConnectorFactory`. | HF-style config/model mappings for composite model, vision tower, connector, and processor creation. |
| Training policy | `training_recipe` owns loading tweaks, tuning policy, and saving. | `train.strategy` owns tuning/PEFT policy; `utils.model_loading` owns loading; `utils.checkpoint` owns resume discovery. |
| Dataset path | JSON loaded eagerly by `LazySupervisedDataset`; text/image preprocessing split across legacy preprocessors/templates. | Hugging Face Dataset pipeline with streaming JSON-array reader, dataset adapters, normalized messages, assistant masks, processor-based encoding. |
| Processor contract | Tokenizer lived on the LLM side, image preprocessing lived on the vision side, and callers composed them manually. | `create_tinyllava_processor` builds the HF processor, injects TinyLLaVA chat-template anchors, aligns the image token, patch count, and vision feature strategy. |
| Trainer | Custom `LLaVATrainer`. | Thin `TinyLlavaTrainer` subclass over `transformers.Trainer`, adding optional modality-length grouping and using `processing_class=processor`. |
| Evaluation | Multiple task-specific generation entrypoints and conversion utilities. | Shared YAML-driven `batch_generation` plus `eval.tasks` loaders/evaluators; benchmark post-processing stays outside the core graph. |
| DeepSpeed/resume | Mostly delegated to runtime/trainer defaults. | Explicit DeepSpeed helper and checkpoint discovery integrated into training entrypoint. |

## Cleanup Notes

The current architecture still contains transitional files that are useful for
compatibility or not yet cleaned up. They should not be treated as anchors for
new work unless they are deliberately promoted back into the active design:

- `tinyllava/model/llm/gemma.py` and similar thin LLM wrappers: legacy registry
  remnants while language models increasingly rely on HF native classes.
- `tinyllava/model/connector/identity.py` and older connector modules:
  compatibility wrappers outside the active HF connector-config path.
- Local generated or scratch content such as `checkpoints/`, `eval_results/`,
  `test.py`, `test_elm.py`, and downloaded model weights under source paths.

For new functionality, prefer extending the active layers shown above:
YAML schema, `utils.model_loading`, `train.strategy`, `data.adapters`,
`eval.tasks`, or the HF-style auto mappings.
