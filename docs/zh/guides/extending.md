# 扩展 TinyLLaVA

模型构建、调优策略、数据转换和评分是独立职责，应使用当前规范接口。

| 扩展目标 | 入口 |
| --- | --- |
| 语言模型 | HF text config、`language_model_name_or_path` |
| 视觉编码器 | `tinyllava.model.vision_tower.auto` 与 image processor 注册 |
| 连接器 | `tinyllava.model.connector.auto` |
| 数据格式 | `tinyllava.data.adapters` |
| 对话模板 | `model.chat_template_path`、Jinja generation block |
| 调优策略 | `tinyllava.train.strategy` |
| 评测任务 | `tinyllava.eval.tasks.auto` |

连接器输出维度必须匹配语言模型，并验证 config/model 保存加载往返。对话模板应正确放置图像占位符，用 `{% generation %}` 标记 assistant 输出；在训练前检查实际渲染结果和监督 mask。

新增数据源应转换为统一消息和图像载荷，数据集专有路径由 adapter 处理。新增公开接口时更新对应 API 页面。参见[架构](../architecture.md)。
