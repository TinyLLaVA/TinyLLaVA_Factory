# 从 v1 迁移到 v2

## 为什么迁移

当前 `main` 分支最初主要作为 TinyLLaVA 论文的轻量研究复现代码库开发。v1 的目标边界是复现论文中的训练与评测流程，实现围绕这些实验展开：训练脚本中写死了 GPU 分配、精度、batch 和输出路径，实验专用分支与未清理代码也随开发积累。

原有输入流程形成时，多模态 processor 与[Jinja Chat Template](https://huggingface.co/docs/transformers/chat_templating_writing)的使用规范尚未像今天这样完善。项目自行维护 formatter 和 template 类，分词与图像预处理分散在不同组件中。现在，Transformers 由 processor 管理多模态对话模板和图像 token 展开，模型提供的格式可以随 checkpoint 保存，并在不同应用中复用。

随着社区模型、checkpoint 格式和训练 API 持续演进，旧接口需要反复增加项目专用适配，已经难以跟上社区节奏。v2 的目的就是将这套研究代码的基础接口对齐到 Hugging Face 模型与 processor 体系，降低持续接入社区能力的维护成本。训练、评测与部署共享加载和生成流程，实验设置则统一记录在 YAML 中。

迁移主要解决三个问题：

- **统一 checkpoint 结构**：用组合配置描述语言模型、视觉编码器和连接器，统一加载与保存接口。
- **统一输入处理**：由一个 processor 管理分词、图像处理、对话模板和图像 token 对齐。
- **明确扩展边界**：分离模型加载、调优策略、数据适配和评测，便于独立扩展各个模块。

## develop 的迁移路线

提交历史中的迁移沿着四个相互衔接的方向展开：

1. **模型基础接口**：[对齐 HF Auto 体系](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/955f5d9)，将 factory 专用加载改为配置/模型映射，引入嵌套组件配置和延迟导入。
2. **模板与输入处理**：[接入 Jinja 模板](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/bf7aa08)，替换项目内模板类；随后[迁移 SFT 数据流](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/ed74649)，将编码和 assistant mask 统一到 processor 路径。
3. **训练配置与策略**：[结构化 YAML](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/c093160)集中管理实验设置，[strategy 插件](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/f2a42bf)将调优策略与模型加载、保存分离。
4. **评测与服务**：[统一批量生成](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/d8b57cc)并[迁移 Web demo](https://github.com/TinyLLaVA/TinyLLaVA_Factory/commit/a599089)，让 benchmark 与交互使用共享 checkpoint 加载和生成接口。

## 接口变化

| v1 | v2 |
| --- | --- |
| 自有语言、视觉和连接器 factory | HF 风格的配置/模型映射与组合模型 |
| training recipe 控制加载和保存 | 模型加载工具、调优策略和 checkpoint 工具 |
| `conv_version` 模板注册表 | 用 `model.chat_template_path` 选择 processor 对话模板 |
| 分离的文本和图像预处理 | processor 编码与 assistant-token mask |
| 各命令分别配置训练参数 | YAML 配置与 dotlist 覆盖 |
| 各任务独立生成 | 共享批量生成接口与任务加载器、评测器 |
| `tinyllava.serve.cli` | `python -m tinyllava.eval.single_turn` |

## 迁移实验

1. 将模型、数据和训练参数分别写入 YAML 对应区域，可从 `configs/train/` 中的配置开始。
2. 预训练设置各组件模型路径；微调将 `model.pretrained_model_name_or_path` 指向预训练 checkpoint。
3. 选择对话模板，用一个小 batch 检查提示词、图像 token 和 assistant 损失 mask。
4. 根据 GPU 数重新计算梯度累积，保持全局 batch 一致。
5. 使用配套评测配置生成答案，并执行对应任务的计分步骤。

Qwen2 base legacy 配方提供论文使用的提示词格式，参见[训练](guides/training.md)与[评测](guides/evaluation.md)。

## 兼容范围与限制

**checkpoint。** 加载器为已知的旧语言模型、视觉编码器和连接器结构注册了权重键映射。这些映射只负责重命名权重，不补齐缺失的配置与 processor 资产。迁移后的 checkpoint 需要匹配的文本、视觉和连接器配置，以及权重、tokenizer、image processor 和 chat template。请验证加载保存往返，并用固定输入比较输出。需要原始运行环境时使用 `legacy/v1`。

**训练状态。** 加载模型权重用于开始新的训练阶段。旧运行环境保存的优化器、调度器与分布式状态，在断点恢复前需要单独验证兼容性。当前恢复流程面向本训练栈保存的完整 checkpoint。

**实验结果。** 这次迁移更新软件接口，论文中的分数仍对应原实验设置。提示词渲染、图像预处理、监督 mask、有效 batch 和生成参数都会影响实验可比性。比较论文结果前，应在相同数据划分与计分口径下重新评测迁移后的 checkpoint。

**扩展代码。** 已移除的 factory 和 recipe 接口需要按新接口适配。可用组件与注册入口见[扩展指南](guides/extending.md)。

## 文档

当前文档随实现一起构建。历史 `doc` 分支保存 v1 的独立 Sphinx 文档，作为归档保留。v2 的配置与命令以本站指南为准。
