# 配置 API

训练与评测共享参数定义。训练参数扩展 HF TrainingArguments，以下列出项目定义的字段。

训练与评测 workflow 统一调用 `utils.config.compose_config` 组合实验 YAML。训练工作进程由 `train/train.py` 读取已解析的 YAML、应用启动器参数并选择算法；评测参数由 `hparams.parser` 解析。

::: tinyllava.hparams.parser
    options:
      members: [parse_eval_arguments, build_eval_arguments]

::: tinyllava.hparams.arguments
    options:
      members: [ModelArguments, DataArguments, TrainingArguments, EvalModelArguments, EvalDataArguments, EvalGenerationArguments, EvalRuntimeArguments, EvalOutputArguments, EvalArguments]

::: tinyllava.hparams.tuning
    options:
      members: [TuningArguments]

::: tinyllava.hparams.training
    options:
      members: [TrainingInputs, build_training_arguments, prepare_training_config]

::: tinyllava.utils.config
    options:
      members: [compose_config, load_config, merge_mappings, resolve_interpolations]

配置层级：

| 字段 | 职责 |
| --- | --- |
| `algorithm` | 训练算法；目前仅 `sft`，默认值也是 `sft` |
| `model` | 模型组件、tokenizer、模板和视觉特征 |
| `data` | 数据来源与处理 |
| `training.tuning` | 参数更新策略、冻结范围、LoRA 参数 |
| `training` | 核心步数、epoch、学习率与 batch |
| `training.saving` | 输出目录、checkpoint 保存与恢复 |
| `training.logging` | 日志、run name、上报后端 |
| `training.optimizer` / `training.scheduler` | 优化器细节与学习率调度 |
| `training.runtime` | 精度、DeepSpeed、梯度检查点、分布式运行参数 |
| `training.dataloader` / `training.evaluation` | 采样加载与训练期评估 |
| `launcher` | 分布式进程、设备与端口；不传入算法参数对象 |

`from` 是 TinyLLaVA 统一的 YAML 组合特性，可用于任意字典节点：单个路径引用一个文件，
路径列表按顺序合并多个文件。优先级依次为：前面的文件、后面的文件、本地字段、命令行覆写。
字典递归合并，列表和标量整体替换；列表内的字典同样支持引用。

不带路径的文件名按节点查找：`model` 对应 `configs/models/`，`connector` 对应
`configs/connectors/`，`training` 对应 `configs/training/`，`data` 则由训练或评测入口
分别选择 `configs/data/train/`、`configs/data/eval/`。阶段内和仓库外的入口同样适用。
预设不存在时直接报错，不回退到同名本地文件。用 `./custom.yaml`、`../custom.yaml`
明确引用本地文件，路径相对于声明它的 YAML；命令行显式相对路径相对于工作目录。
支持绝对路径和 `~`。没有预设目录的其他节点，其 `from` 仍按相对文件路径解析。

只引用文件时，`model: phi` 等价于 `model: {from: phi}`；`connector`、
`training`、`data` 同样支持，省略扩展名时自动补 `.yaml`，也可显式写 `.yaml` 或 `.yml`。
显式路径同样适用，例如 `./custom` 引用当前文件旁的 `custom.yaml`。
命令行可写 `model=phi model.model_max_length=4096`，引用与字段覆写会同时保留。
普通数据、模板与 checkpoint 路径字段仍保持原值。`from` 使用字面路径，组合后移除；
普通字段仍支持 `${dataset_dir}`、`${oc.env:NAME,default}` 等 OmegaConf 插值，在运行时默认值补全后解析。
缺失文件、非字典文件、非法引用与循环引用会报错，同一文件可以在不同位置复用。

通过 `model.from=phi` 或 `model.connector.from=mof`
选择文件；统一用 `key=value` 新增或覆盖字段。切换引用保留显式同级字段，不残留被替换文件的默认值。
实验名由 `experiment_name` 或入口文件名决定。阶段 checkpoint 依赖使用 `from_stage`。
训练预设根部直接存放训练字段，由 `training.from` 引用。训练与评测数据分别位于
`configs/data/train/`、`configs/data/eval/`，都直接存放数据字段，由 `data.from` 引用。
例如 `training: {from: [pretrain, ./hardware]}` 可以组合多份训练配置。

```yaml
# configs/experiments/custom.yaml
algorithm: sft
model:
  from: phi
  model_max_length: 4096
  connector:
    from: mlp2x_gelu
    depth: 3
    bias: false
training:
  from: lora_finetune
  per_device_train_batch_size: 4
  global_batch_size: 32
  learning_rate: 0.0002
  saving:
    output_dir: output/phi/sft
    save_steps: 500
  logging:
    logging_steps: 10
    report_to: tensorboard
  tuning:
    strategy: lora
    tune_type_llm: lora
    tune_type_vision_tower: frozen
    tune_type_connector: full
    peft:
      r: 128
      lora_alpha: 256
launcher:
  distributed: true
  devices: 0,1
```

通用 Trainer 参数适配器将以上 batch 设置换算为 `gradient_accumulation_steps: 4`。
`global_batch_size` 显式设置时覆盖已有的梯度累积值，换算后从工作进程配置中移除。

`utils/config.py:compose_config` 是唯一的文件组合入口。工作进程通过 `load_config`
读取已经组合好的 YAML，不再展开引用。`train/workflow.py` 只处理共享配置与阶段的
继承、checkpoint 和启动：阶段值覆盖共享值，包括通过阶段引用引入的值。

`hparams/training.py` 统一定义字段分组、配置校验、batch 换算和 HF 参数适配；
`hparams/tuning.py` 定义参数更新策略的字段。保存、日志、优化器等规则与算法无关。
SFT 位于 `train/algorithms/sft.py`，复用这些通用能力，只负责监督训练执行；
采样与 Trainer 扩展位于 `train/algorithms/sft_trainer.py`。参数更新策略仍在 `train/strategy/`。
新算法可扩展配置段或替换准备/执行逻辑，无需复制通用字段表。

当前仅实现 SFT。算法调用通用参数适配器时得到
`TrainingInputs(model, data, training, tuning)`；RL 可以使用自己的输入对象。

评测仍使用 `model.model_name_or_path` 指定 checkpoint，`model.model_id` 标记预测结果；
评测可使用 `model: ./eval-model.yaml` 引用包含评测模型字段的 YAML；checkpoint 路径
写在 `model.model_name_or_path` 中。

所有 `configs/experiments/` 入口都显式写出 `algorithm: sft`；
解析器对省略算法的自定义配置仍默认使用 SFT。`experiment_name` 只控制默认输出目录
和日志名，不选择模型。用 `--steps` 执行单个阶段，也可以提供省略 `stages` 的单阶段入口。
训练预设不改变模型的长度限制或注意力实现。

只有构建 HF `TrainingArguments` 时，通用适配器才将 saving、logging 等分组展开；
YAML 和 dry-run 始终保留分组，tuning 单独构建为 `TuningArguments` 供策略使用。
