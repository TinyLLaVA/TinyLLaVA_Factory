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


## Example
This section we provide an example of finetuning our TinyLLaVA-3.1B checkpoint on the [pokemon dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions).

First, use this script to download and convert the pokemon dataset.

**Replace the path to your dataset path**

```python
import shortuuid
from datasets import load_dataset
from PIL import Image
import random
import json
import tqdm
import os

ds = load_dataset('lambdalabs/pokemon-blip-captions')
pokemon_data = []

pokemon_image_path = '/path/to/your/data/pokemon/image'
pokemon_data_path = '/path/to/your/pokemon_blip_captions.json'

description_list = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented."
]

for sample in tqdm.tqdm(ds['train']):
    uuid = shortuuid.uuid()
    sample_dict = dict()
    sample_dict['id'] = uuid
    sample_dict['image'] = 'pokemon/image/' + uuid + '.jpg'
    sample['image'].save(os.path.join(pokemon_image_path, uuid + '.jpg'))
    conversations = [
        {"from": "human", "value": "<image>\n" + random.choice(description_list)},
        {"from": "gpt", "value": sample['text']}
    ]
    sample_dict['conversations'] = conversations
    pokemon_data.append(sample_dict)

with open(pokemon_data_path, 'w') as f:
    json.dump(pokemon_data, f, indent=4)
```

Now, after we have acquired the dataset following our pretraining format, we can now use lora to finetune our model!

**Replace the path to your dataset path**
```shell
#!/bin/bash

# Assign the arguments to variables
DATA_PATH="/path/to/your/pokemon_blip_captions.json"
IMAGE_PATH="/path/to/your/data/"
OUTPUT_DIR="/path/to/your/TinyLLaVA-3.1B-lora"

deepspeed tinyllava/train/train.py \
    --deepspeed ./scripts/tiny_llava/zero3.json \
    --lora_enable True --lora_r 32 --lora_alpha 64 \
    --model_name_or_path bczhou/TinyLLaVA-3.1B \
    --version phi \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_PATH\
    --vision_tower bczhou/TinyLLaVA-3.1B-SigLIP \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --fp16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to wandb \
    
```
**Note: set `group_by_modality_length` to `False` if our data only contains image-text pairs.**

After this step, you should be able to use the LoRA finetuned model on your downstream tasks.