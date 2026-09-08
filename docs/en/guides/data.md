# Training data

The base recipe uses LLaVA's 558K image-caption pretraining set followed by
the 665K instruction mixture.

Download annotations and images following the dataset maintainers' instructions:

- [LLaVA pretraining data](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
- [LLaVA instruction data](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
- [ShareGPT4V data preparation](https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md)

Preserve the relative paths encoded in each annotation. A typical LLaVA layout is:

```text
dataset/
  text_files/
    blip_laion_cc_sbu_558k.json
    llava_v1_5_mix665k.json
  llava/llava_pretrain/images/
  coco/train2017/
  gqa/images/
  ocr_vqa/images/
  textvqa/train_images/
  vg/VG_100K/
  vg/VG_100K_2/
```

Pretraining uses `dataset/llava/llava_pretrain/images` as the image root;
the instruction mixture uses `dataset` because its paths include dataset prefixes.
For example, `coco/train2017/example.jpg` resolves relative to `data.image_folder`.

## Legacy annotations

Select `data.dataset_adapter=llava_legacy` for a JSON array with this structure:

```json
[
  {
    "id": "example",
    "image": "coco/train2017/example.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nDescribe this image."},
      {"from": "gpt", "value": "A person riding a bicycle."}
    ]
  }
]
```

Omit `image` for text-only samples. The adapter converts `from`/`value` into
`role`/`content`; the reader builds an Arrow cache incrementally, and the
processor resolves images and encodes samples lazily. Assistant masks determine
which tokens contribute to the training loss.

## Custom data

The native pipeline normalizes message records before processor encoding.
Use an adapter to map a different on-disk schema into these records. See the [data API](../reference/data.md) and
[extension guide](extending.md). Image paths, conversation roles, and the chat
template must be validated together before launching a full training run.
