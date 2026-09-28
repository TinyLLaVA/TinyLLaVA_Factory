# 训练数据

预训练常用 LLaVA 558K 图文对，微调常用 LLaVA 665K 对话。下载入口：[LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)、[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)。微调标注引用多个数据集的图像，需一并下载并按标注路径组织。

## 默认目录与数据集名称

数据默认存放在 `datasets/`，标注位于 `datasets/text_files/`，评测输入位于 `datasets/eval/`。保留原标注中的图片相对路径，例如 `datasets/coco/train2017/`、`datasets/gqa/images/`、`datasets/vg/VG_100K/`。预训练图片根目录为 `datasets/llava/llava_pretrain/images/`，665K 和 ShareGPT4V 的图片根目录为 `datasets/`。

`configs/data/` 统一定义数据集，实验通过 `dataset: llava_558k`、`llava_665k`、`sharegpt4v_pretrain` 或 `sharegpt4v_finetune` 选择。添加一个 YAML 文件即可注册新名称。用 `dataset_dir=/path/to/data` 或 `TINYLLAVA_DATASET_DIR` 整体移动数据根目录。checkpoint、预测和计分结果统一写入 `output/`。

本机已从 `/data/vlm/llava_data` 硬链接 1,489,338 个文件，未链接 checkpoint 和已有评测结果。目录重新建立，文件共享原 inode；编辑标注或图片前先复制，避免修改共享源内容。LLaVA 558K、665K 和清洗后的 ShareGPT4V 微调集图片引用完整；ShareGPT4V 预训练集缺少 543,842 个图片引用，需补齐后再运行。完整本地统计见仓库 `datasets/README.md`。

## 数据入口

`data.dataset_name_or_path` 接受 Hub 数据集 ID、本地数据集目录、JSON/JSONL/Parquet 文件或 `parquet` 等 Hugging Face builder。原配置项 `data.data_path` 已替换，已有 YAML 和命令行覆盖项需同步改名。

```yaml
data:
  dataset_name_or_path: mvp-lab/LLaVA-OneVision-1.5-Instruct-Data
  dataset_config_name: CLEVR
  split: train[:1%]
  revision: main
  cache_dir: .cache/datasets
  dataset_adapter: auto
```

本地分片可以通过 builder 和 `data_files` 指定，支持路径、列表、通配符及 split 到文件的映射：

```yaml
data:
  dataset_name_or_path: parquet
  data_files:
    train: /datasets/instruct/train-*.parquet
    validation: /datasets/instruct/validation-*.parquet
  split: train
  dataset_adapter: auto
```

`dataset_config_name` 对应 HF 的 `name`（子集）；`data_dir`、`revision`、`cache_dir` 传给上游 loader。直接指定单个文件时，其 split 为 `train`，可以使用 `train[:10%]` 等切片；此时不要再指定 `data_files`、`data_dir`、`dataset_config_name` 或 `revision`。需要这些选项时使用 builder 或目录入口。

直接传入的本地 JSON 数组由增量 reader 读取；通过 Hub、目录或 builder 选择的数组由上游 loader 读取，内存行为也由上游决定。训练使用可随机访问的 Arrow Dataset，以支持索引和模态分组，目前不提供 iterable streaming 训练。

文件格式加载与样本字段适配分开处理。原生样本需要 `messages` 或 `conversations`，其中图像占位符与 `image`/`images` 载荷对应。支持 PIL 图像、多图列表及 Arrow 的 `{bytes, path}` 图像值。FineVision 的 `texts` 等其他结构仍需 dataset adapter。

`auto` 对使用图像路径的 LLaVA 记录应用 legacy adapter，对原生消息和内嵌媒体保留原始结构。已有 legacy 训练配置显式设置了 `llava_legacy`；切换原生数据时应将它改为 `auto`。

JSON 数组的缓存指纹由文件路径、大小、修改时间、reader 实现、解析库版本及 adapter 实现/状态共同决定，不再手工维护 `json-array-vN`。缓存标识与发布版本无关。如果外部工具在替换文件时同时保留大小和修改时间，需更新文件时间或清除对应缓存。

## Legacy 标注

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

`data.dataset_name_or_path` 指向标注，`data.image_folder` 是图像路径的根目录。例如记录中的 `coco/train2017/example.jpg` 应相对这个根目录定位，因此图像根目录应设为 `coco` 的上一级目录。

标注支持 JSON 数组和 JSONL。JSON 数组由 `ijson` 的 YAJL C 后端增量读取，该后端在常见平台由预编译 wheel 提供。允许文件开头带 UTF-8 BOM；数组元素必须为对象，并使用标准 JSON 语法，缺失或多余的逗号、截断内容、`NaN` 和 `Infinity` 均会被拒绝。JSONL 通过 Hugging Face JSON loader 加载。

adapter 将原始记录转换为统一消息和图像输入，processor 负责编码，collator 负责 padding 和监督标签。训练前检查图像路径、模板渲染和 assistant-token mask，确认损失计算覆盖预期的回答 token。

详见[数据 API](../reference/data.md)与[训练指南](training.md)。
