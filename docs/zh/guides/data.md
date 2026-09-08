# 训练数据

预训练常用 LLaVA 558K 图文对，微调常用 LLaVA 665K 对话。下载入口：[LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)、[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)。微调标注引用多个数据集的图像，需一并下载并按标注路径组织。

legacy adapter 接受类似以下的 JSON 记录：

```json
{
  "id": "example",
  "image": "coco/train2017/example.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe this image."},
    {"from": "gpt", "value": "A description of the image."}
  ]
}
```

`data.data_path` 指向标注，`data.image_folder` 是图像路径的根目录。例如记录中的 `coco/train2017/example.jpg` 应相对这个根目录定位，因此图像根目录应设为 `coco` 的上一级目录。

adapter 将原始记录转换为统一消息和图像输入，processor 负责编码，collator 负责 padding 和监督标签。训练前检查图像路径、模板渲染和 assistant-token mask，确认损失计算覆盖预期的回答 token。

详见[数据 API](../reference/data.md)与[训练指南](training.md)。
