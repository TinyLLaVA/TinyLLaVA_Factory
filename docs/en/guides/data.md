# Training data

The base recipe uses LLaVA's 558K image-caption pretraining set followed by
the 665K instruction mixture.

Download annotations and images following the dataset maintainers' instructions:

- [LLaVA pretraining data](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
- [LLaVA instruction data](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
- [ShareGPT4V data preparation](https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md)

Preserve the relative paths encoded in each annotation. A typical LLaVA layout is:

```text
datasets/
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

Pretraining uses `datasets/llava/llava_pretrain/images` as the image root;
the instruction mixture uses `datasets` because its paths include dataset prefixes.
For example, `coco/train2017/example.jpg` resolves relative to `data.image_folder`.

## Named datasets and storage

Select a dataset with `dataset: llava_558k` or `dataset: llava_665k`. Their annotation
and image paths are defined once in `configs/data/`. ShareGPT4V presets are
`sharegpt4v_pretrain` and `sharegpt4v_finetune`. Add a YAML file there to register
another dataset; its filename is the selector name.

`dataset_dir` defaults to `datasets/`; override it per run or set
`TINYLLAVA_DATASET_DIR`. Evaluation inputs live under `datasets/eval/`.
Keep checkpoints and predictions under `output/`, outside the input tree.
Hard-linking files from a shared collection on the same filesystem avoids copying
image payloads; recreate directories and preserve relative paths. Linked files
share contents with their sources, so copy a file before editing it.

The local collection linked from `/data/vlm/llava_data` contains complete image
references for LLaVA 558K, LLaVA 665K and the cleaned ShareGPT4V fine-tuning set.
The ShareGPT4V pretraining annotation references 543,842 unavailable images;
complete those downloads before running its stage. See `datasets/README.md` in
the repository for the local validation inventory.

## Data sources

Set `data.dataset_name_or_path` to a Hub dataset ID, a local dataset directory,
a local JSON/JSONL/Parquet file, or a Hugging Face builder such as `parquet`.
This replaces the former `data.data_path` setting; update existing YAML files
and command-line overrides to the new name.

```yaml
data:
  dataset_name_or_path: mvp-lab/LLaVA-OneVision-1.5-Instruct-Data
  dataset_config_name: CLEVR
  split: train[:1%]
  revision: main
  cache_dir: .cache/datasets
  dataset_adapter: auto
```

For local shards, use a builder and `data_files`. Paths, lists, glob patterns,
and split-to-file mappings follow `datasets.load_dataset`:

```yaml
data:
  dataset_name_or_path: parquet
  data_files:
    train: /datasets/instruct/train-*.parquet
    validation: /datasets/instruct/validation-*.parquet
  split: train
  dataset_adapter: auto
```

`dataset_config_name` maps to HF's `name` (subset); `data_dir`, `revision`, and
`cache_dir` are forwarded to the loader. A direct file defines a `train` split
and accepts slices such as `train[:10%]`. For a direct file, omit `data_files`,
`data_dir`, `dataset_config_name`, and `revision`; use a builder or directory
when these options are needed. Local JSON arrays passed directly are parsed
incrementally; arrays selected through a Hub repository, directory, or builder
use the upstream loader and its memory behavior.

Training uses an indexed Arrow Dataset for random access and modality grouping.
It does not expose HF iterable streaming. Opening a file format does not infer
a dataset's conversation schema: native rows need `messages` or `conversations`,
with image placeholders matching their `image`/`images` payloads. Decoded PIL
images, image lists, and Arrow `{bytes, path}` image values are supported.
Schemas such as FineVision's `texts` need a dataset adapter.

`auto` applies the legacy adapter to path-based LLaVA records; it preserves
native messages and embedded media. Existing legacy training recipes explicitly
set `llava_legacy`; change that setting to `auto` when switching to native data.

The JSON-array cache fingerprint includes the resolved path, file size, mtime,
reader implementation, parser version, and adapter implementation/state. These
are internal cache identities, not release versions. No manual `json-array-vN`
sequence is used. If external tooling replaces a file while preserving both size
and mtime, touch the file or remove its cache before loading it again.

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

JSON arrays are read incrementally with the `ijson` YAJL C backend, supplied
by its wheels on common platforms. An optional UTF-8 BOM is accepted. Arrays
must contain objects and use standard JSON syntax: missing or trailing commas,
truncated input, `NaN`, and `Infinity` are rejected. JSONL annotations are also
supported through the Hugging Face JSON loader.

## Custom data

The native pipeline normalizes message records before processor encoding.
Use an adapter to map a different on-disk schema into these records. See the [data API](../reference/data.md) and
[extension guide](extending.md). Image paths, conversation roles, and the chat
template must be validated together before launching a full training run.
