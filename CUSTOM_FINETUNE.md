# Finetune TinyLLaVA with Custom Datasets

Given the needs of finetuning with custom datasets, we provide a tutorial on how to custom finetune on our trained model, e.g. tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B (HF path).

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

<details>
You can use the following scripts to convert the Pokemon dataset to the above data format.
<summary>converting data format</summary>
  
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

</details>

## Custom Finetune
After acquiring the dataset following the above data format, you can finetune our trained model TinyLLaVA-Phi-2-SigLIP-3.1B checkpoint by using lora.

- Replace data paths and `output_dir` with yours in `scripts/train/custom_finetune.sh`
- Adjust your GPU ids (localhost) and `per_device_train_batch_size` in `scripts/train/custom_finetune.sh`.

```bash
bash scripts/train/custom_finetune.sh
```

## Evaluation with Custom Finetuned Model
All of the models trained by TinyLLaVA Factory have the same evaluation procedure, no matter it is trained through custom finetune or through normal training. Please see the [Evaluation](https://tinyllava-factory.readthedocs.io/en/latest/Evaluation.html) section in our Doc.



