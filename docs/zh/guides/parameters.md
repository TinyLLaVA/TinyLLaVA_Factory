# 参数速查

下文训练字段的路径相对于 stage。例如，多阶段实验用
`stages.finetune.training.learning_rate=1e-5` 修改学习率；评测字段直接位于 YAML 根部。
实际值由所选预设和覆写决定，用 `--dry-run` 查看解析后的设置。
文件引用和组合规则见[配置方式](configuration.md)。

## 模型 { #model }

以下字段放在 `model` 下。组件预设与 checkpoint 的选择见[模型指南](models.md)。

### 来源与加载

| 字段 | 用途 |
| --- | --- |
| `pretrained_model_name_or_path` | 加载完整 TinyLLaVA 或 PEFT checkpoint。 |
| `language_model_name_or_path` | 组装组件时加载的语言模型。 |
| `vision_model_name_or_path` | 组装组件时加载的视觉编码器。 |
| `connector` | 连接器预设或配置字典，例如 `mlp2x_gelu`。 |
| `cache_dir` | Hugging Face 模型下载缓存目录。 |
| `attn_implementation` | 注意力后端，例如 `sdpa`；省略时使用模型默认值。 |

### 分词

| 字段 | 用途 |
| --- | --- |
| `tokenizer_name_or_path` | 覆写 checkpoint 或语言模型自带的 tokenizer。 |
| `model_max_length` | 训练序列长度上限，例如 `2048`。 |
| `tokenizer_use_fast` | 请求 fast tokenizer，用于 token 与字符位置对齐。 |
| `tokenizer_padding_side` | tokenizer 的 padding 方向，训练通常使用 `right`。 |

### 对话与视觉特征

| 字段 | 用途 |
| --- | --- |
| `chat_template` | 内联 Jinja 对话模板。 |
| `chat_template_path` | Jinja 模板文件，与 `chat_template` 二选一。 |
| `vision_feature_layer` | 隐藏层索引或索引列表，例如 `-2`。 |
| `vision_feature_select_strategy` | `full` 保留全部 token，`default` 去掉首个；省略时沿用 checkpoint 设置或按视觉塔识别。 |

## 训练数据 { #training-data }

以下字段放在阶段的 `data` 下。目录组织和样本格式见[训练数据](data.md)。

### 来源选择

| 字段 | 用途 |
| --- | --- |
| `dataset_name_or_path` | Hub ID、本地目录或文件，或 `parquet` 等 builder。 |
| `dataset_config_name` | 数据子集名称，例如 `CLEVR`。 |
| `revision` | Hub 分支、标签或提交。 |
| `split` | 数据划分或切片，例如 `train`、`train[:1%]`。 |

### 文件与样本格式

| 字段 | 用途 |
| --- | --- |
| `data_files` | 传给 builder 的文件、通配符或 split 到文件的映射。 |
| `data_dir` | 传给数据 loader 的子目录。 |
| `cache_dir` | 下载与 Arrow 缓存目录。 |
| `image_folder` | 标注中图片相对路径的根目录。 |
| `dataset_adapter` | `auto` 识别已知格式；`llava_legacy` 选择 LLaVA 对话记录。 |

## 训练 { #training }

以下字段直接放在 `training` 下。起始配方见[训练指南](training.md#recipes)。

### Batch 与时长

| 字段 | 用途 |
| --- | --- |
| `per_device_train_batch_size` | 每个设备单次前向的样本数。 |
| `gradient_accumulation_steps` | 每次优化器更新前累积的前向、反向次数。 |
| `global_batch_size` | 按设备数和单设备 batch 推导梯度累积，覆盖显式累积值。 |
| `learning_rate` | 初始学习率。 |
| `num_train_epochs` | 遍历训练数据的轮数。 |
| `max_steps` | 正值按优化器步数控制训练时长，优先于 epoch 数。 |

全局 batch = worker 数 × 单设备 batch × 梯度累积步数。
目标全局 batch 必须能被 worker 数 × 单设备 batch 整除。

### 优化器与学习率调度

| 字段 | 用途 |
| --- | --- |
| `optim` | 当前 Transformers 版本支持的优化器。 |
| `weight_decay` | 权重衰减系数。 |
| `lr_scheduler_type` | 调度方式，例如 `cosine`。 |
| `lr_scheduler_kwargs` | 调度器的附加参数字典。 |
| `warmup_steps` | 预热步数，或小于 `1` 的比例，例如 `0.03`。 |

### 精度、显存与加载

| 字段 | 用途 |
| --- | --- |
| `precision` | `auto`、`fp32`、`fp16` 或 `bf16`，据此派生 HF 精度标志。 |
| `gradient_checkpointing` | 重算激活以减少训练显存占用。 |
| `deepspeed` | DeepSpeed JSON 路径或配置字典。 |
| `dataloader_num_workers` | 数据加载进程数。 |
| `remove_unused_columns` | 保持 `false`，保留 collator 使用的多模态字段。 |
| `train_sampling_strategy` | `group_by_length` 按长度采样。 |
| `group_by_modality_length` | 按模态与长度组织多模态、纯文本样本。 |
| `eval_strategy` | 内置 SFT 配方使用 `no`，当前 SFT 数据模块不提供验证集。 |

### 保存与恢复

| 字段 | 用途 |
| --- | --- |
| `output_dir` | 阶段 checkpoint 目录，通常由实验名与阶段名生成。 |
| `save_strategy` | 按 `steps`、`epoch` 保存，或设为 `no`。 |
| `save_steps` | 按步保存时的间隔。 |
| `save_total_limit` | 保留的 checkpoint 数量。 |
| `save_only_model` | 只保存模型，不保存优化器状态；需要完整恢复时保持 `false`。 |
| `resume_from_checkpoint` | `auto` 查找 `output_dir` 中最近的完整 checkpoint，也可指定路径。 |

### 日志

| 字段 | 用途 |
| --- | --- |
| `logging_steps` | 日志间隔。 |
| `report_to` | 日志后端，例如 `tensorboard` 或 `none`。 |
| `run_name` | 运行名称，通常为 `<实验名>/<阶段名>`。 |

## 组件调优 { #tuning }

以下字段放在 `training.tuning` 下。

| 字段 | 用途 |
| --- | --- |
| `strategy` | `common` 普通更新，`lora` 适配器训练，`lora_int8` 在 8-bit 语言模型上训练适配器。 |
| `tune_type_llm` | 语言模型更新方式，按策略支持 `frozen`、`full`、`lora`。 |
| `tune_type_vision_tower` | 视觉塔更新方式，包括 `frozen`、`full`、`partially-tune`。 |
| `tune_vision_tower_from_layer` | 部分训练视觉塔时的首个可训练层。 |
| `tune_type_connector` | 连接器更新方式，内置配方使用 `full`。 |
| `peft` | 上游 LoRA 参数，例如 `{r: 128, lora_alpha: 256}`。 |

## 阶段与启动 { #stages }

### 实验与阶段选择

| 字段 | 用途 |
| --- | --- |
| 根部 `experiment_name` | 实验名，默认取入口文件名并去掉扩展名。 |
| 根部 `output_root` | 实验输出的父目录，默认 `output`。 |
| 根部 `dataset_dir` | 输入根目录，默认取 `TINYLLAVA_DATASET_DIR` 或 `datasets`。 |
| 阶段内 `algorithm` | 训练算法，目前为 `sft`。 |
| 阶段内 `checkpoint` | 初始 checkpoint，`null` 表示从组件组装；后续阶段通常使用上一阶段输出。 |
| 阶段内 `from_stage` | 指定用于初始化的更早阶段。 |
| 单阶段入口的 `stage` | 输出路径中使用的阶段名。 |

### 启动器

以下字段放在 `launcher` 下，可在实验根部共享，也可按阶段覆写。

| 字段 | 用途 |
| --- | --- |
| `distributed` | 启用 torchrun。 |
| `devices` | GPU 编号，例如 `[0, 1]`；`auto` 读取 `CUDA_VISIBLE_DEVICES`。 |
| `master_port` | 分布式通信端口，默认 `29501`。 |

## 评测 { #evaluation }

以下字段全部位于 YAML 根部。评测数据准备和计分见[评测指南](evaluation.md)。

### 模型与数据

| 字段 | 用途 |
| --- | --- |
| `model_name_or_path` | TinyLLaVA checkpoint 目录或 Hub 仓库，评测流程也接受 `MODEL_PATH` 环境变量。 |
| `model_id` | 写入预测结果的模型标识，默认取 `MODEL_NAME` 或 checkpoint 名称。 |
| `adapter` | 输入 loader，例如 `vqa`、`scienceqa`、`mmmu`，由数据预设提供。 |
| `question_file` | 评测标注文件。 |
| `image_folder` | 图片相对路径的根目录。 |

### 生成

| 字段 | 用途 |
| --- | --- |
| `temperature` | 非正值关闭采样，贪心生成可设为 `0.0`。 |
| `top_p` | 核采样的概率阈值。 |
| `num_beams` | 解码束数。 |
| `max_new_tokens` | 每条回答最多生成的 token 数。 |
| `chat_template` | 内联 Jinja 模板覆写。 |
| `chat_template_path` | Jinja 文件覆写，与 `chat_template` 二选一。 |

### Worker 与输出

| 字段 | 用途 |
| --- | --- |
| `device` | Worker 设备，省略或设为 `null` 时优先 CUDA，否则 CPU。 |
| `batch_size` | 每次生成的对话数。 |
| `num_chunks` | 直接启动生成 worker 时的分片数。 |
| `chunk_idx` | 直接启动 worker 时的分片索引，从零开始。 |
| `answers_file` | 预测 JSONL 文件，评测流程中默认 `${output_dir}/answers.jsonl`。 |

### 评测流程

| 字段 | 用途 |
| --- | --- |
| `benchmark` | 评测流程名称，例如 `gqa`、`textvqa`。 |
| `dataset_dir` | 输入根目录，默认取 `TINYLLAVA_DATASET_DIR` 或 `datasets`。 |
| `eval_dir` | 评测资源目录，默认取 `EVAL_DIR` 或 `${dataset_dir}/eval`。 |
| `output_dir` | 预测、格式转换和计分结果的目录。 |
| `shard` | 在选定的每张 GPU 上启动一个生成 worker，随后合并预测。 |
| `devices` | 分片生成使用的 GPU 编号，例如 `[0, 1]`。 |

`shard: true` 时，流程自动分配 worker 设备、分片索引和临时预测路径。
评测 loader 提供任务专用提示，包括 ScienceQA 的选项字母回答要求。
交互推理的 CLI 选项见[推理指南](inference.md#options)。
