# 扩展 TinyLLaVA

模型构建、调优策略、数据转换和评分是独立职责，应使用当前规范接口。

| 扩展目标 | 入口 |
| --- | --- |
| 语言模型 | HF text config、`language_model_name_or_path` |
| 视觉编码器 | HF Auto 注册、懒加载 `AutoVisionTowerModel` 和 `AutoImageProcessor` |
| 连接器 | `tinyllava.model.connector.auto` |
| 数据格式 | `tinyllava.data.adapters` |
| 对话模板 | `model.chat_template_path`、Jinja generation block |
| 调优策略 | `tinyllava.train.strategy` |
| 评测任务 | `tinyllava.eval.tasks.auto` |

连接器输出维度必须匹配语言模型，并验证 config/model 保存加载往返。对话模板应正确放置图像占位符，用 `{% generation %}` 标记 assistant 输出；在训练前检查实际渲染结果和监督 mask。

新增数据源应转换为统一消息和图像载荷，数据集专有路径由 adapter 处理。新增公开接口时更新对应 API 页面。参见[架构](../architecture.md)。

## 视觉模型与 I-MoF

原生 CLIP、SigLIP、DINOv2 及接口兼容的 HF 视觉模型通过 `AutoVisionTowerModel` 查询 HF 当前的 Auto 映射。
CLIP/SigLIP 的完整配置会先提取视觉子配置。新增模型使用 HF 公开的 Auto 注册接口，
返回空间 token 的 hidden states，并提供匹配的 `hidden_size`、`patch_size` 和图像
token 元数据。预处理类由 checkpoint 保存的配置交给 `AutoImageProcessor` 解析。

MoF 来自 Tong 等人的 [Eyes Wide Shut? Exploring the Visual Shortcomings of
Multimodal LLMs](https://arxiv.org/abs/2401.06209)（CVPR 2024）；
[TinyLLaVA Factory 表 2](https://arxiv.org/html/2405.11788v1) 明确引用了该方法。
上游 `mof_mlp.py` 实现的是 **I-MoF**：CLIP 与 DINOv2 分别经过独立 MLP，之后按空间
位置交错排列 token。迁移保留这一语义，以及两路共用 CLIP 预处理图像的原有约定。
A-MoF 则在投影前混合特征，两者不同。

- `MofVisionConfig` 保存两个分支配置，`MofVisionModel` 支持标准构造与保存加载，
  构造函数不下载模型。
- 两路 patch size、通道数和深度必须相同，隐藏维度可以不同。DINOv2 的位置编码
  插值到 CLIP 输入分辨率。
- `MofConnector` 独立投影后输出 `clip_0, dino_0, clip_1, dino_1, ...`。
  同时选择多个隐藏层时，各层特征仍按分支分别送入投影。
- `TinyLlavaProcessor` 通过 connector 配置的 `get_output_sequence_length`
  获取占位符数量。MoF 将 N 个选中特征映射为 2N 个输出 token，处理器不按 MoF 名称分派。
- 所有 connector 都接收 `vision_hidden_size`、`text_hidden_size` 和 `vision_feature_layer`。
  各实现负责输入校验，输出长度计算委托给 connector 配置。通用模型没有 MoF 分支，
  也不根据视觉模型自动指定 connector。

先将两个预训练分支组装为可复用的视觉 checkpoint：

```python
from transformers import AutoImageProcessor
from tinyllava.model.vision_tower.mof import MofVisionModel

clip_path = "openai/clip-vit-large-patch14-336"
vision = MofVisionModel.from_pretrained_components(
    clip_model_name_or_path=clip_path,
    dinov2_model_name_or_path="facebook/dinov2-large",
)
vision.save_pretrained("output/mof-vision")
AutoImageProcessor.from_pretrained(clip_path).save_pretrained("output/mof-vision")
```

MoF 在自己的 connector 配置中用 `vision_hidden_sizes` 声明拼接特征的两个宽度
（依次为 CLIP、DINOv2），并校验总和等于 `vision_hidden_size`。
普通 connector 只接收输入宽度，不依赖视觉模型配置。更换分支时应同步配置两个宽度。

通过现有配置入口显式选择视觉塔和对应 connector：

```bash
python -m tinyllava.run --config configs/train/pretrain.yaml \
  model.vision_model_name_or_path=output/mof-vision \
  model.connector_config=configs/connectors/mof.yaml model.vision_feature_select_strategy=default \
  name=qwen2_mof
```

完整 checkpoint 保存两路视觉权重及投影。使用 `AutoVisionTowerModel` 按需加载
独立 MoF checkpoint。导入包或配置不会加载模型实现，只有选中 MoF 时才解析对应模块。
直接使用 HF `AutoModel` 加载 MoF 时，先显式导入 `MofVisionModel` 完成类注册。
使用 `tinyllava.data.processor.AutoProcessor.from_pretrained` 懒加载已保存的 processor；
直接使用 HF `AutoProcessor` 加载自定义 processor 时，先显式导入 `TinyLlavaProcessor` 完成注册。本地测试验证小模型数值行为和保存加载往返，
论文 benchmark 分数仍需训练与评测验证。

## 自定义 image processor

`tinyllava.data.image_processor` 保留为训练和推理共用的懒加载扩展入口，原生处理器
继续由 HF 解析。新增项目实现时，例如在 `tinyllava/data/image_processor/custom/`
导出 `CustomImageProcessor`，并在 `image_processor/auto/auto_mappings.py` 的
`CUSTOM_IMAGE_PROCESSOR_MAPPING_NAMES` 中加入 `("custom", "CustomImageProcessor")`。
使用唯一类名，通过 `save_pretrained` 将其保存为 `image_processor_type`。
只有 checkpoint 选中的实现才会被导入，本地 Auto 类不修改 HF 全局映射或加载方法。

## Connector 实现

每个 connector 包包含 `configuration_<name>.py`、`modeling_<name>.py`
和懒加载 `__init__.py`，旧装饰器注册和单文件实现已移除。线性投影使用
`MLPConnectorConfig(depth=1)`；Identity 校验拼接后的视觉宽度与语言模型宽度一致。

Resampler 使用 Perceiver 可学习查询；Q-Former 使用 HF 维护的
`Blip2QFormerModel`，按配置初始化，不下载 BERT。二者每张图输出
`num_queries` 个 token。所有 TinyLLaVA connector 使用同一个
`TinyLlavaProcessor`，保存 connector 配置并调用其长度方法。单独加载处理器
只需配置类，不需要导入 connector 模型实现或构造权重。
迁移后的架构需要重新训练 connector，不支持旧权重布局；参数冻结由训练策略控制。

processor Auto 映射以复合模型配置 `TinyLlavaConfig` 为键，与 HF 的模型/处理器
关系一致。新增 connector 的长度规则只需在其配置中实现
`get_output_sequence_length(input_length)`，不增加 processor 子类或映射项。
MLP/Identity 保持 N，MoF 返回 2N，查询型 connector 返回配置的固定数量。
模型侧特征拆分调用同一个方法，避免维护两份长度规则。

## 语言模型

`AutoLanguageModel` 和 `AutoLanguageModelForCausalLM` 实时使用 HF 的
backbone/causal-LM 映射。原生 Llama、Gemma、Phi、Qwen2、StableLM
不需要项目封装；导入后通过 HF 公共接口注册的模型也能被识别。
项目映射只描述本地实现，按需加载选中的模块；配置注册不会导入模型实现或加载权重。

保留 OpenELM，是因为项目已有对应预设，而当前 Transformers 依赖尚未提供原生实现。
规范实现位于 `tinyllava.model.llm.openelm`，使用 HF Cache、GenerationMixin、
标准嵌入访问接口及共享输出头。本地支持官方 270M 的共享嵌入配置，保留层参数名称，
移除了未使用的旧 backbone classifier。随包保留
[Apple 许可证](https://huggingface.co/apple/OpenELM-270M-Instruct/blob/main/LICENSE)。

通过项目 Auto 入口按需加载 OpenELM。独立使用 HF
`AutoModel`/`AutoModelForCausalLM` 时，先显式导入对应 OpenELM 类以激活公共注册。
复合模型通过 `get_input_embeddings()` 推导共享权重路径，不假设语言模型内部属性名称。

验证覆盖微型离线检查点、有/无缓存解码、左侧 padding、多模态训练及保存恢复；
尚未进行完整预训练 OpenELM 的效果评估。
