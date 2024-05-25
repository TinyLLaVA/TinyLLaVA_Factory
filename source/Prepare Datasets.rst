Prepare Datasets
====================

As reported in our paper_, we use two different datasets: the LLaVA_dataset_ and the ShareGPT4V_dataset_. In this section, we will detail data preparation for training. For evaluation dataset, please see instructions in the Evaluation section

.. _paper: https://arxiv.org/abs/2405.11788
.. _LLaVA_dataset: https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#pretrain-feature-alignment
.. _ShareGPT4V_dataset: https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md

LLaVA dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

• **Pretraining Images**: The pretraining images of LLaVA is from the 558K subset of the LAION-CC-SBU dataset. Download as follows.

  • LAION-CC-SBU-558K: images.zip_.

  .. _images.zip: https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip

• **Pretraining Annotations**: The pretraining annotations of LLaVA. Download as follows.

  • pretraining annotations: blip_laion_cc_sbu_558k.json_.

  .. _blip_laion_cc_sbu_558k.json: https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain

• **SFT Images**: The SFT images of LLaVA. Download as follows.

  • LAION-CC-SBU-558K: Already download as "LAION-CC-SBU-558K" in Pretraining Images.

  • COCO: This dataset is from the COCO2017_challenge_. Download: train2017_.

  .. _COCO2017_challenge: https://cocodataset.org/
  .. _train2017: http://images.cocodataset.org/zips/train2017.zip

  • GQA: GQA_project_page_. Download: gqa_images_.

  .. _GQA_project_page: https://cs.stanford.edu/people/dorarad/gqa/about.html
  .. _gqa_images: https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip

  • OCR-VQA: OCR-VQA_project_page_. Download: download_script_. We save all files as ``.jpg``.

  .. _OCR-VQA_project_page: https://ocr-vqa.github.io/
  .. _download_script: https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_

  • TextVQA: TextVQA_project_page_. Download: trainval_images_.

  .. _TextVQA_project_page: https://textvqa.org/
  .. _trainval_images: https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip

  • VisualGenome: VisualGenome_project_page_. Download: part1_, part2_.

  .. _VisualGenome_project_page: https://homes.cs.washington.edu/~ranjay/visualgenome/index.html
  .. _part1: https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
  .. _part2: https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip


• **SFT Annotations**: The SFT annotations of LLaVA. Download as follows.

  • SFT annotations: llava_v1_5_mix665k.json_.

  .. _llava_v1_5_mix665k.json: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json

ShareGPT4V dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• **Pretraining and SFT Images**: The images of ShareGPT4V. Download as follows.

  • LAION-CC-SBU-558K: Already download as "LAION-CC-SBU-558K" in LLaVA's Pretraining Images.

  • COCO: Already download as "COCO" in LLaVA's SFT Images.
  
  • WebData & Share_TextVQA: This dataset is curated by the ShareGPT4V_project_. Download: images_. Only for academic usage.

  .. _ShareGPT4V_project: https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V
  .. _images: https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax

  • SAM: This dataset is collected by Meta. Download: sam_images_. We only use 000000~000050.tar for now. If you just want to use ShareGPT4V for SFT, you can quickly download 9K_images_.

  .. _sam_images: https://ai.meta.com/datasets/segment-anything-downloads/
  .. _9K_images: https://drive.google.com/file/d/1dKumdOKSXtV7lIXdrG7jsIK_z2vZv2gs/view?usp=drive_link

  • GQA: Already download as "GQA" in LLaVA's SFT Images.

  • OCR-VQA: Already download as "OCR-VQA" in LLaVA's SFT Images.

  • TextVQA: Already download as "TextVQA" in LLaVA's SFT Images.

  • VisualGenome: Already download as "VisualGenome" in LLaVA's SFT Images.

• **Pretraining Annotations**: The pretraining annotations of ShareGPT4V. Download as follows.

  • pretraining annotations: share-captioner_coco_lcs_sam_1246k_1107.json_.

  .. _share-captioner_coco_lcs_sam_1246k_1107.json: https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/share-captioner_coco_lcs_sam_1246k_1107.json

• **SFT Annotations**: The SFT annotations of ShareGPT4V. Download as follows.

  • SFT annotations: sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json_.

  .. _sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json: https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json


Organize Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Organize the image files and annotation files as follows in ``path/to/your/dataset`` :

.. code-block:: bash

   dataset
   ├── llava
   │   ├── llava_pretrain
   │   │   ├── images
   ├── coco
   │   ├── train2017
   ├── sam
   │   ├── images
   ├── gqa
   │   ├── images
   ├── ocr_vqa
   │   ├── images
   ├── textvqa
   │   ├── train_images
   ├── vg
   │   ├── VG_100K
   │   ├── VG_100K_2
   ├── share_textvqa
   │   ├── images
   ├── web-celebrity
   │   ├── images
   ├── web-landmark
   │   ├── images
   ├── wikiart
   │   ├── images
   ├── text_files
   │   ├── blip_laion_cc_sbu_558k.json
   │   ├── llava_v1_5_mix665k.json
   │   ├── share-captioner_coco_lcs_sam_1246k_1107.json
   │   ├── sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json
