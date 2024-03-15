# Finetune TinyLLaVA on Custom Datasets

## Dataset Format

Convert your data to a JSON file of a List of all samples. Sample metadata should contain `id` (a unique identifier), `image` (the path to the image), and `conversations` (the conversation data between human and AI).

Here's an example of the [pokemon dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) turned into the data format:

```json
[
  {
        "id": "meiKqU2auAVK2vrtLhKGoJ",
        "image": "pokemon/image/meiKqU2auAVK2vrtLhKGoJ.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nProvide a brief description of the given image."
            },
            {
                "from": "gpt",
                "value": "a drawing of a green pokemon with red eyes"
            }
        ]
    }
]
```

## Custom Finetune
If you have limited task-specific data, we recommend finetuning our checkpoints with LoRA using this [script](https://github.com/DLCV-BUAA/TinyLLaVABench/tree/main/scripts/tiny_llava/finetune/finetune_lora.sh).
