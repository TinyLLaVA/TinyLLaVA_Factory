# Parameter reference

Training paths below are relative to a stage. For example, use
`stages.finetune.training.learning_rate=1e-5` in a multi-stage experiment.
Evaluation fields go directly at the YAML root. Actual values come from the selected
presets and overrides; use `--dry-run` to inspect the resolved configuration.
See [Configuration](configuration.md) for file references and composition.

## Model { #model }

Set these fields under `model`. For component presets and checkpoint selection,
see [Models and checkpoints](models.md).

### Sources and loading

| Field | Use |
| --- | --- |
| `pretrained_model_name_or_path` | Load a complete TinyLLaVA or PEFT checkpoint. |
| `language_model_name_or_path` | Language model to load when assembling components. |
| `vision_model_name_or_path` | Vision encoder to load when assembling components. |
| `connector` | Connector preset or mapping, e.g. `mlp2x_gelu`. |
| `cache_dir` | Hugging Face model download cache. |
| `attn_implementation` | Attention backend, e.g. `sdpa`; omit to use the model's default. |

### Tokenization

| Field | Use |
| --- | --- |
| `tokenizer_name_or_path` | Override the tokenizer supplied by the checkpoint or language model. |
| `model_max_length` | Training sequence length limit, e.g. `2048`. |
| `tokenizer_use_fast` | Request a fast tokenizer; needed for token-to-character alignment. |
| `tokenizer_padding_side` | Tokenizer padding side; training normally uses `right`. |

### Conversation and vision

| Field | Use |
| --- | --- |
| `chat_template` | Inline Jinja conversation template. |
| `chat_template_path` | Jinja template file; choose this or `chat_template`. |
| `vision_feature_layer` | Hidden layer index or list of indices, e.g. `-2`. |
| `vision_feature_select_strategy` | `full` keeps all tokens; `default` drops the first. Omit to preserve checkpoint settings or infer the backbone's behavior. |

## Training data { #training-data }

Set these fields under the stage's `data`. See [Training data](data.md) for file layouts and sample formats.

### Source selection

| Field | Use |
| --- | --- |
| `dataset_name_or_path` | Hub ID, local directory/file, or builder such as `parquet`. |
| `dataset_config_name` | Dataset subset name, e.g. `CLEVR`. |
| `revision` | Hub branch, tag, or commit. |
| `split` | Split or slice, e.g. `train` or `train[:1%]`. |

### Files and sample format

| Field | Use |
| --- | --- |
| `data_files` | Files, glob patterns, or a split-to-files mapping for a dataset builder. |
| `data_dir` | Data subdirectory passed to the dataset loader. |
| `cache_dir` | Download and Arrow cache directory. |
| `image_folder` | Root for relative image paths in annotations. |
| `dataset_adapter` | `auto` detects known formats; `llava_legacy` selects LLaVA conversation records. |

## Training { #training }

Set these fields directly under `training`. Choose a starting recipe in the
[training guide](training.md#recipes).

### Batch size and duration

| Field | Use |
| --- | --- |
| `per_device_train_batch_size` | Samples per device per forward pass. |
| `gradient_accumulation_steps` | Forward/backward passes per optimizer update. |
| `global_batch_size` | Derive accumulation from device count and per-device batch; overrides explicit accumulation. |
| `learning_rate` | Initial learning rate. |
| `num_train_epochs` | Number of passes through the training data. |
| `max_steps` | Positive values set the optimizer-step budget instead of the epoch budget. |

Global batch = worker count × per-device batch × accumulation steps.
The requested global batch must be divisible by worker count × per-device batch.

### Optimizer and schedule

| Field | Use |
| --- | --- |
| `optim` | Optimizer supported by the installed Transformers version. |
| `weight_decay` | Weight decay coefficient. |
| `lr_scheduler_type` | Schedule, e.g. `cosine`. |
| `lr_scheduler_kwargs` | Additional scheduler options as a mapping. |
| `warmup_steps` | A step count, or a fraction below `1`, e.g. `0.03`. |

### Precision, memory, and loading

| Field | Use |
| --- | --- |
| `precision` | `auto`, `fp32`, `fp16`, or `bf16`; derives HF's precision flags. |
| `gradient_checkpointing` | Recompute activations to reduce training memory. |
| `deepspeed` | DeepSpeed JSON path or configuration mapping. |
| `dataloader_num_workers` | Data-loading worker count. |
| `remove_unused_columns` | Keep `false` for the multimodal fields consumed by the collator. |
| `train_sampling_strategy` | Use `group_by_length` for length-based sampling. |
| `group_by_modality_length` | Group multimodal and text-only samples by length. |
| `eval_strategy` | Shipped SFT recipes use `no`; the current SFT data module supplies no validation dataset. |

### Saving and resuming

| Field | Use |
| --- | --- |
| `output_dir` | Stage checkpoint directory; normally derived from experiment and stage names. |
| `save_strategy` | Save by `steps`, `epoch`, or `no`. |
| `save_steps` | Save interval when using step-based saving. |
| `save_total_limit` | Number of checkpoints to retain. |
| `save_only_model` | Save model weights without optimizer state; keep `false` for full training resume. |
| `resume_from_checkpoint` | `auto` finds the last complete checkpoint in `output_dir`; a path selects one explicitly. |

### Logging

| Field | Use |
| --- | --- |
| `logging_steps` | Logging interval. |
| `report_to` | Reporting backend, e.g. `tensorboard` or `none`. |
| `run_name` | Run name; normally `<experiment>/<stage>`. |

## Component tuning { #tuning }

Set these fields under `training.tuning`.

| Field | Use |
| --- | --- |
| `strategy` | `common` for ordinary updates, `lora` for adapters, `lora_int8` for adapters with an 8-bit language model. |
| `tune_type_llm` | Language-model policy: `frozen`, `full`, or `lora`, depending on the strategy. |
| `tune_type_vision_tower` | Vision policy, including `frozen`, `full`, and `partially-tune`. |
| `tune_vision_tower_from_layer` | First trainable vision layer for partial tuning. |
| `tune_type_connector` | Connector update policy; shipped recipes use `full`. |
| `peft` | Upstream LoRA options, e.g. `{r: 128, lora_alpha: 256}`. |

## Stages and launch { #stages }

### Experiment and stage selection

| Field | Use |
| --- | --- |
| Root `experiment_name` | Experiment name; defaults to the entry filename without its extension. |
| Root `output_root` | Parent of experiment output directories; defaults to `output`. |
| Root `dataset_dir` | Input root; defaults to `TINYLLAVA_DATASET_DIR` or `datasets`. |
| Stage `algorithm` | Training algorithm; currently `sft`. |
| Stage `checkpoint` | Initial checkpoint; `null` starts from components. Later stages normally use the preceding output. |
| Stage `from_stage` | Initialize from a named earlier stage. |
| Single-stage `stage` | Stage name used in the output path. |

### Launcher

Set these fields under `launcher`, shared at the experiment root or overridden per stage.

| Field | Use |
| --- | --- |
| `distributed` | Enable torchrun. |
| `devices` | GPU IDs, e.g. `[0, 1]`; `auto` reads `CUDA_VISIBLE_DEVICES`. |
| `master_port` | Distributed rendezvous port; default `29501`. |

## Evaluation { #evaluation }

All fields below live at the YAML root. See [Evaluation](evaluation.md) for benchmark assets and scoring.

### Model and data

| Field | Use |
| --- | --- |
| `model_name_or_path` | TinyLLaVA checkpoint directory or Hub repository; the workflow also accepts `MODEL_PATH`. |
| `model_id` | Identifier written to predictions; defaults to `MODEL_NAME` or the checkpoint name. |
| `adapter` | Benchmark input loader, e.g. `vqa`, `scienceqa`, or `mmmu`; supplied by input presets. |
| `question_file` | Benchmark annotations. |
| `image_folder` | Root for relative image paths. |

### Generation

| Field | Use |
| --- | --- |
| `temperature` | Nonpositive values disable sampling; use `0.0` for greedy benchmark generation. |
| `top_p` | Nucleus sampling cutoff. |
| `num_beams` | Number of decoding beams. |
| `max_new_tokens` | Generated-token limit per answer. |
| `chat_template` | Inline Jinja template override. |
| `chat_template_path` | Jinja file override; choose this or `chat_template`. |

### Workers and output

| Field | Use |
| --- | --- |
| `device` | Worker device; omitted or `null` selects CUDA when available, otherwise CPU. |
| `batch_size` | Conversations per generation batch. |
| `num_chunks` | Shard count when running generation workers directly. |
| `chunk_idx` | Zero-based shard index for a direct worker. |
| `answers_file` | Prediction JSONL path; the workflow defaults to `${output_dir}/answers.jsonl`. |

### Benchmark workflow

| Field | Use |
| --- | --- |
| `benchmark` | Benchmark workflow, e.g. `gqa` or `textvqa`. |
| `dataset_dir` | Shared input root; defaults to `TINYLLAVA_DATASET_DIR` or `datasets`. |
| `eval_dir` | Evaluation assets; defaults to `EVAL_DIR` or `${dataset_dir}/eval`. |
| `output_dir` | Directory for predictions, conversions, and scores. |
| `shard` | Run a generation worker per selected GPU, then merge predictions. |
| `devices` | GPU IDs for sharded generation, e.g. `[0, 1]`. |

With `shard: true`, the workflow assigns worker devices, shard indices, and temporary
answer paths. Benchmark loaders supply task-specific prompts, including ScienceQA's
option-letter instruction. Interactive inference uses CLI options listed in the
[inference guide](inference.md#options).
