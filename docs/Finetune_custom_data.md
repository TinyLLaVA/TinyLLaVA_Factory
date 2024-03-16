# Finetune LLaVA on Custom Datasets

## Dataset Prepare

Convert your data to a JSON file of a List of all samples. Sample metadata should contain `id` (a unique identifier), `image` (the path to the image), and `conversations` (the conversation data between human and AI).

A sample JSON for finetuning LLaVA for generating tag-style captions for Stable Diffusion:

```json
[
  {
    "id": "997bb945-628d-4724-b370-b84de974a19f",
    "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
      },
      {
        "from": "gpt",
        "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
      }
    ]
  },
  ...
]
```

## Finetune

You can finetune TinyLLaVA with Phi is possible with one 24GB GPU using the `finetune_lora.sh`. 

If you want to full finetune TinyLLaVA with your dataset, you can use the `finetune.sh`. 

The `.sh` file you can found in the path `script/tiny_llava` .


