# v2 路线图

v2 路线图包括发布验证、DeepStack Hook 和原生 Qwen3-VL 接入。
下表列出当前状态和下一验收目标。已有实验的更新步骤见[迁移指南](migration.md)。

| 工作方向 | 状态 | 下一验收目标 |
| --- | --- | --- |
| 模型、processor 与实验接口 | v2 开发分支已实现 | 完成代表性训练、checkpoint 重载、生成与评测验证 |
| Qwen3-VL 风格 DeepStack Hook 插件 | 独立原型 | 为每次调用绑定输入，并保持到 checkpoint 重计算结束 |
| 原生 Qwen3-VL 与动态分辨率输入 | 计划中 | 端到端对齐 packed patches、媒体网格、占位符、位置编码与组批 |

## Qwen3-VL / DeepStack Hook 插件 { #deepstack-hooks }

[DeepStack](https://arxiv.org/abs/2406.04334)沿网络深度分配视觉信息，
[Qwen3-VL](https://arxiv.org/abs/2511.21631)为此使用多个视觉层的特征。
拟议插件将这一思路用于组装的 HF 语言模型：通过 decoder 层的 `register_forward_hook`，
在层输出的视觉位置加上已对齐的视觉特征。变换保持 `[batch, sequence, hidden]` 形状，
沿用宿主模型的 decoder 循环、注意力、位置编码与缓存实现。

### 职责划分

| 部分 | 职责 |
| --- | --- |
| 视觉编码器与连接器 | 产生主视觉特征和辅助特征，对齐维度、媒体与 token 顺序 |
| 组合 VLM | 选择注入层，将特征映射到本次调用的位置，处理生成切片与 batch 扩展 |
| 层适配器 | 定位原生 decoder 层，更新 hidden states 并保留 Tensor 或 tuple 输出结构 |
| 调用上下文 | 管理每次调用的特征和 mask，覆盖 forward 与 backward 重计算 |
| Checkpoint 配置 | 描述扩展，在加载时恢复 hook 与可训练参数 |

Auto 映射继续负责选择语言模型类，插件为选中的实例附加层级行为。
序列长度、注意力拓扑和多模态位置编码的变更由具体架构的模型适配器负责。

### 原型证据

本地原型已完成以下验证。
保存的快照使用 Torch 2.13.0、Transformers 5.13.1、CPU/FP32，以及随机初始化的微型
Llama、Qwen3、Gemma2、OpenELM。视觉输入是模拟 connector 输出的可求梯度张量。
插件目前处于原型阶段，训练与生成接口的集成按下列里程碑推进。

| 检查 | 四类模型的快照结果 |
| --- | --- |
| 零注入 | logits 与原模型完全一致 |
| 非零注入 | logits 改变，视觉特征梯度非零 |
| Reentrant / non-reentrant checkpoint | logits、视觉梯度和模型参数梯度与普通执行一致 |
| 缓存单 token 解码 | 与整段执行一致 |
| 参数名称与 hook 移除 | 参数键名保持不变，移除后恢复原始输出 |
| Backward 前移除 hook | 重计算时复现 `CheckpointError` |

**Hook 及其调用输入必须保持到 backward 重计算结束。**
原型使用覆盖整个 backward 的串行上下文；下一步需要在多个 forward 等待反向传播时，
仍能为每次重计算取回对应的输入。

### 集成里程碑

1. **调用归属。** 将特征和 mask 绑定到每次调用，验证两个 forward 后再 backward、
   梯度累积与两种 checkpoint 模式。
2. **可训练特征链路。** 连接可训练 connector，冻结语言模型，对比普通执行下的 connector 梯度。
3. **生成与输出。** 验证有/无 cache 的 `generate`、beam 扩展、padding 和多图。
   根据本次调用实际包含的视觉位置决定注入，并明确捕获 hook、`output_hidden_states`
   观察注入值时的顺序与语义。
4. **保存与重载。** 恢复扩展配置与可训练权重，保持宿主参数名称，重载后验证零注入等价性。
5. **执行后端。** 基于已建立的训练与生成检查，验证混合精度、分布式训练和编译执行。

运行时集成采用共享的层输出变换，用小型适配器处理层定位与输出结构。
公开配置字段在调用生命周期与序列化方案验证后确定。

## 原生 Qwen3-VL 与动态分辨率 { #native-qwen3-vl }

原生 Qwen3-VL 接入为组件和 DeepStack 实验提供端到端基线。
其 processor、packed-patch 布局、`image_grid_thw`、模态 token ID、spatial merger、
位置编码与视频时间戳，需要在预处理、组批、forward 和生成中保持对齐。

视觉特征交接应明确所选输出、隐藏维度、逐媒体长度，以及已完成的合并和投影。
主视觉特征与 DeepStack 辅助特征分别传到组合模型中的对应位置。
`ComponentProcessorMixin` 已提供可复用的配置和长度方法，各原生 processor 决定这些方法
在自身处理流程中的调用位置。

验收从混合分辨率图像 batch 和图文混合样本开始，再扩展到多图和视频。
在固定输入上对比集成路径与原生模型，覆盖缓存生成和 checkpoint 重载。
