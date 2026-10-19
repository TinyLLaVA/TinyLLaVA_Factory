# 架构

TinyLLaVA Factory 将模型组合、输入处理、训练策略和评测分别组织为独立模块。训练、批量评测与交互推理共享模型加载和 processor 接口。

## 模型组件

`TinyLlavaConfig` 描述三个组件：

- `text_config`：语言模型。
- `vision_config`：视觉编码器。
- `connector_config`：将视觉特征映射到语言模型嵌入空间的连接器。

`TinyLlavaForConditionalGeneration` 提供 `forward` 和 `generate`。使用 `from_pretrained` 加载完整 checkpoint，或用 `from_pretrained_components` 组装初始组件。

processor 组合 tokenizer、image processor 与 chat template，负责对齐图像占位符与视觉 token，并准备模型输入。

## 训练数据流

```mermaid
flowchart LR
    records[标注与图像] --> adapter[数据适配器]
    adapter --> messages[消息与图像载荷]
    messages --> processor[Processor]
    processor --> collator[Collator 与标签]
    collator --> trainer[Trainer]
    model[组合模型] --> trainer
    strategy[调优策略] --> model
    trainer --> checkpoint[checkpoint]
```

adapter 统一数据格式，processor 编码文本与图像，collator 补齐序列并根据 assistant mask 创建损失标签。调优策略选择可训练参数并配置 PEFT，Trainer 负责优化过程和 checkpoint 保存。

## 评测与推理

checkpoint 加载器返回模型及其 processor。生成助手编码对话、调用模型并解码回答。评测任务提供输入加载器和对应的计分或导出步骤。

数据集评测使用 `tinyllava.eval.batch_generation`，命令行推理使用 `tinyllava.eval.single_turn`，Web UI 使用 `tinyllava.serve.app`。

## CLI 与执行职责

`tinyllava.cli` 分发到训练、评测或对话入口。`utils/config.py` 展开递归 `from` 引用，
使用 OmegaConf 合并配置。训练和评测 workflow 处理阶段继承、checkpoint、路径与执行步骤。
多阶段实验在各 stage 内选择算法和数据，根部共享模型、训练默认值和启动设置。

`hparams/arguments.py` 定义训练与评测的项目参数对象；`hparams/training.py` 负责训练结构校验、
batch 推导和参数构造。`hparams/hf.py` 包含 HF `TrainingArguments` 子类，在构造训练 worker
输入时导入，保持 workflow 校验轻量。workflow 解析路径并移除编排字段后，
`EvalArguments(**config)` 校验扁平评测参数。

`train/algorithms/sft.py` 执行监督训练，
`train/algorithms/sft_trainer.py` 提供 SFT 的 Trainer 与采样扩展；
`train/strategy/` 提供冻结、LoRA 等策略。其他算法与 SFT 同属 `algorithms/`，
可复用通用参数设施并替换自己的数据、batch 语义和训练循环。
`runner.py` 执行命令、管理进程和合并输出。评测任务各自定义数据适配、转换和评分步骤。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `tinyllava.model` | 组合配置、模型、视觉编码器和连接器 |
| `tinyllava.data` | 数据适配器、processor 创建、模板与 collator |
| `tinyllava.train` | 训练入口、调优策略和采样 |
| `tinyllava.eval` | 生成、任务加载器与评测器 |
| `tinyllava.hparams` | 参数类型、配置校验与 HF 参数适配 |
| `tinyllava.utils` | YAML 组合、checkpoint 发现和模型加载 |
| `tinyllava.serve` | 交互式 Web UI |

实现细节见[扩展指南](guides/extending.md)和[API 参考](reference/model.md)。

[v2 路线图](roadmap.md#deepstack-hooks)介绍拟议的 Qwen3-VL 风格 DeepStack Hook 插件：
视觉组件产生辅助特征，组合模型将其映射到视觉位置，层适配器向原生 HF decoder 注入特征。
已有独立原型为后续集成提供验证依据。
