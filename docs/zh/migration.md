# 迁移到 v2

v2 保留 TinyLLaVA 的视觉编码器、连接器和语言模型组合方式，为实验提供共享的
模型/processor 接口、可组合 YAML，以及明确的训练和评测流程。本页说明当前开发接口的
迁移方法；DeepStack Hook 与原生 Qwen3-VL 接入计划见 [v2 路线图](roadmap.md)。

YAML 示例见[配置方式](guides/configuration.md)，字段含义见[参数速查](guides/parameters.md)，
组件与权重选择见[模型指南](guides/models.md)。

## 命令与配置

通过 `tinyllava` 或 `python -m tinyllava.cli` 执行任务：

| 任务 | 命令 |
| --- | --- |
| 训练 | `python -m tinyllava.cli train --config configs/experiments/phi.yaml` |
| 评测 | `python -m tinyllava.cli eval --config configs/eval/gqa.yaml model_name_or_path=output/phi/finetune` |
| 单轮推理 | `python -m tinyllava.cli single-turn --model-path output/phi/finetune --query "Describe the image." --image-file image.jpg` |
| 多轮对话 | `python -m tinyllava.cli chat --model-path output/phi/finetune` |

迁移 v1 shell 配方时，将模型、数据、训练与启动设置写入实验 YAML。
已有的早期 v2 开发配置按下表更新：

| 原设置 | 当前设置 |
| --- | --- |
| `python -m tinyllava.run` | `python -m tinyllava.cli` 加任务子命令 |
| 通过 `model.name` 选择模型 | `model: phi` 或 `model: {from: phi, ...}` |
| `from_config`、`from_yaml` 或 Hydra `defaults` 文件引用 | 目标节点的 `from`，接受路径或有序路径列表 |
| 实验 YAML 中的 `model.connector_config` | `model.connector`，同样支持 `from` 与覆写 |
| `configs/recipes/` | `configs/training/`，引用后字段直接进入 `training` |
| `configs/train/` 入口 | 实验加 `--steps`，或单阶段 YAML |
| 根部 `name` | `experiment_name` |
| 多阶段实验根部的 `algorithm`、`data` | `stages.<name>.algorithm`、`stages.<name>.data` |
| `data.data_path` | `data.dataset_name_or_path` |
| 用 `from` 表示阶段依赖 | `from_stage`；`from` 用于 YAML 组合 |
| `training.saving.*`、`training.logging.*` 等 Trainer 字段分组 | 直接使用 `training.*`，保留 `training.tuning` 及其 `peft` 字典 |
| 评测的 `model.*`、`data.*`、`generation.*`、`runtime.*`、`output.*` | 同名字段直接放到评测 YAML 根部 |
| 评测的 `data.from` | 根部 `from`，路径指向 `configs/data/eval/` 中的文件 |
| `single_pred_prompt` | 删除该选项，ScienceQA loader 提供选项字母回答指令 |

以上 YAML 调整针对实验配置。保存的 HF 模型配置继续使用 `text_config`、`vision_config`
和 `connector_config`。
[v2 发布日志](https://github.com/TinyLLaVA/TinyLLaVA_Factory/blob/develop/CHANGELOG.md)
记录破坏性变更，加载器按当前 schema 校验配置。

### 训练示例

```yaml
# configs/experiments/custom.yaml
model: phi
experiment_name: phi_trial

stages:
  pretrain:
    algorithm: sft
    data: llava_558k
    training: pretrain

  finetune:
    algorithm: sft
    data: llava_665k
    training:
      from: lora_finetune
      learning_rate: 0.0001
      save_steps: 500
      logging_steps: 10
      tuning:
        peft:
          r: 64
          lora_alpha: 128
```

模型、训练和启动器默认值可以在根部共享，每个阶段选择自己的算法与数据。
后续阶段默认加载上一阶段 checkpoint，`from_stage` 选择更早的阶段，`checkpoint`
选择外部来源。单独运行后续阶段前，先准备其依赖的 checkpoint。

单阶段入口将 `algorithm`、`data`、`model`、`training` 放在根部，用 `stage` 命名输出阶段。
训练数据预设位于 `configs/data/train/`，模型与连接器预设分别位于 `configs/models/`
和 `configs/connectors/`。

### 评测示例

```yaml
# configs/eval/custom.yaml
from: ../data/eval/gqa.yaml
benchmark: gqa
model_name_or_path: output/phi_trial/finetune
batch_size: 4
temperature: 0.0
output_dir: output/phi_trial/finetune/eval/gqa
```

评测流程解析路径、按需分片，并将预测写入 `answers_file`，通常为
`${output_dir}/answers.jsonl`。直接启动 `tinyllava.eval.batch_generation` worker 时，
传入已经解析的根部字段，包括 `model_name_or_path` 和 `question_file`。

## Trainer 与调优设置

支持的 Transformers 范围为 `>=5.10.0,<6`。调整 YAML 层级时，同时更新这些 Trainer 选项：

| 原设置 | 当前设置 |
| --- | --- |
| `warmup_ratio: 0.03` | `warmup_steps: 0.03`，小于 `1` 表示比例，大于等于 `1` 表示步数 |
| 独立的 `group_by_length: true` | `train_sampling_strategy: group_by_length` |
| 独立的 `bf16` / `fp16` 标志 | `precision: auto`、`bf16`、`fp16` 或 `fp32` |
| `training.runtime.precision` | `training.precision` |
| `training.saving.resume_from_checkpoint` | `training.resume_from_checkpoint` |

删除过时的 HF 字段：`overwrite_output_dir`、`save_safetensors`、`logging_dir`、
`include_tokens_per_second`、`half_precision_backend`、`use_mps_device`、
`fsdp_transformer_layer_cls_to_wrap`。

`training.global_batch_size` 根据 worker 数和单设备 batch 推导梯度累积，
`training.tuning` 选择冻结与适配器策略。模型长度与注意力后端放在 `model` 下；
内置 Phi LoRA/QLoRA 实验沿用模型预设的 2048 长度和默认注意力后端。

## Python 扩展接口

| 职责 | 当前接口 |
| --- | --- |
| YAML 组合 | `tinyllava.utils.config.compose_config` |
| 阶段依赖与评测路径 | `tinyllava.train.workflow`、`tinyllava.eval.workflow` |
| 项目参数对象 | `tinyllava.hparams.arguments`，包含扁平的 `EvalArguments` |
| HF Trainer 扩展 | `tinyllava.hparams.hf.TrainingArguments` |
| Trainer 输入集合 | `TrainingArgBundle`，替代 `TrainingInputs` |
| 训练参数构造 | `tinyllava.hparams.training.build_training_arguments` |
| 训练算法 | `tinyllava.train.algorithms` 中的模块 |
| 组件更新策略 | `tinyllava.train.strategy` |
| Processor 选择与加载 | `tinyllava.data.processor.AutoProcessor` |
| 可复用的 processor 配置与长度规则 | `ComponentProcessorMixin` |
| LLaVA 处理与从模型创建 processor | `TinyLlavaProcessor` |

更新从已移除的 `configuration.py`、`utils.arguments`、`hparams.parser`、
`hparams.tuning` 模块导入的代码。评测通过 `EvalArguments(**resolved_worker_config)`
构造参数，原先按组拆分的评测参数类已合并。数据 loader 接收任务输入，ScienceQA 提示词由
对应 loader 处理。

将 `data.processor.processing_base` 和 `data.processor.creation` 的导入改为上述 processor
接口。`AutoProcessor.from_config(config, **kwargs)` 根据模型配置选择类，将 `kwargs`
传入其构造函数。通过模型组件创建 LLaVA processor 时使用 `TinyLlavaProcessor.from_model`。
Connector 实现已按包组织，使用 config/model Auto 映射；使用旧装饰器注册或扁平模块的扩展需更新。

通过各组件的 Auto 接口注册配置类与模型类。项目模型类型使用 `mlp__tlf_connector`
等命名空间，选中组件时加载对应实现。新组件需要满足隐藏维度、特征布局、processor 和
生成契约；注册与长度方法见[扩展指南](guides/extending.md)。

## Checkpoint 与输入行为

将模型与 tokenizer、image processor、对话模板一起保存和重载。显式模板覆写优先于
已保存模板。Phi、OpenELM 预设选择内置 LLaVA 模板，Qwen2 base legacy 实验选择其历史模板。
`pretrain_legacy` 是省略用户问题的图像到答案格式，用于对应的预训练配方。

`vision_feature_select_strategy` 可以省略。新组装模型按视觉配置推导，加载 checkpoint
时沿用保存值。SigLIP 使用 `full`，CLIP、DINO、MoF 使用 `default` 去掉首个 CLS token；
显式值优先。

加载器重映射已知的旧语言模型、视觉编码器和 MLP 权重键，迁移时需提供匹配的配置和
预处理资产。重构后的查询型连接器需要匹配当前架构的权重。用 `checkpoint` 初始化新阶段，
用 `training.resume_from_checkpoint` 恢复完整 checkpoint 的训练状态。

原始 v1 实验可通过独立环境中的 `legacy/v1` 使用，历史 `doc` 分支保留 v1 Sphinx 文档。

## 迁移验证

1. **记录基线。** 保存原始环境、组件与数据版本、模板、训练和生成设置，以及固定样本的输出。
2. **预览配置。** 用 `--dry-run` 检查解析后的预设、阶段输入、输出路径和有效全局 batch。
3. **检查处理后样本。** 查看用户问题、图像占位符、assistant mask、截断、视觉特征选择和连接器输出长度。
4. **重载 checkpoint。** 保存并重载模型与 processor，查明缺失或多余权重的原因，在相同精度下对比固定输入的输出。
5. **运行短训练。** 检查可训练参数、有限梯度、loss、断点恢复和阶段衔接。
6. **对照评测。** 对齐任务定义与计分方法，记录数据、模板、预处理、生成预算和资源使用的变化。

复现报告应对应匹配的实验设置；效率报告分别测量启动、训练和生成表现。
