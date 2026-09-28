# Local datasets

`datasets/` is the default input root. Dataset definitions live in `configs/data/`;
training experiments select them by name. Keep downloaded data out of Git.
Use `dataset_dir=/other/root` or `TINYLLAVA_DATASET_DIR` to move the whole collection.

```text
datasets/
  text_files/
    blip_laion_cc_sbu_558k.json
    llava_v1_5_mix665k.json
    really_cleaned_share-captioner_coco_lcs_sam_1246k_1107.json
    cleaned_sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json
  llava/llava_pretrain/images/
  coco/train2017/
  gqa/images/
  ocr_vqa/images/
  textvqa/train_images/
  vg/VG_100K/
  vg/VG_100K_2/
  sam/images/
  share_textvqa/images/
  web-celebrity/images/
  web-landmark/images/
  wikiart/images/
  eval/                       # benchmark inputs and official evaluation tools
```

Preserve annotation-relative image paths. LLaVA 558K uses
`datasets/llava/llava_pretrain/images` as its image root; instruction/ShareGPT4V
annotations use `datasets` itself. Training outputs and evaluation predictions
belong under `output/`, separate from this input tree.

On this machine, 1,489,338 files were hard-linked from `/data/vlm/llava_data` on
2026-09-18. Directories are recreated; regular files share the source inode.
Source checkpoints, prior evaluation answers/results, caches and logs were excluded.
Hard links share file contents: create a separate copy before editing annotations
or images. Removing a local link does not remove the source pathname.

Full annotation/image-path checks produced:

| Preset | Records | Image records | Missing image references |
| --- | ---: | ---: | ---: |
| `llava_558k` | 558,128 | 558,128 | 0 |
| `llava_665k` | 665,298 | 624,610 | 0 |
| `sharegpt4v_pretrain` | 1,229,257 | 1,229,257 | 543,842 |
| `sharegpt4v_finetune` | 665,058 | 624,370 | 0 |

The ShareGPT4V pretraining source is incomplete, including missing SAM images.
Download its referenced images before running that stage; no records were silently
removed. These checks verify path existence, not image decoding or annotation quality.
Machine-local counts are recorded in `.local-links.json` and `.local-validation.json`.
