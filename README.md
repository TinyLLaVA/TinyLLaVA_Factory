# TinyLLaVA Visualizer

TinyLLaVA Visualizer is a specialized visualization tool designed to work with the TinyLLaVA model, a multimodal large model. This tool enables users to visualize the relationships between generated words, their connections to the input image, and the probability distributions of these words during the model's inference process.

## Features

TinyLLaVA Visualizer provides three main visualization functionalities:

1. **Word Relationships**: Visualize the relationships between each generated word and the words generated before it. This allows users to understand how the model builds up context over time.
2. **Word-Image Relationships**: Visualize the relationship between each generated word and the input image. This feature helps users see how the model links textual output to visual input.
3. **Word Probability Distributions**: Visualize the probability distribution of each word during the generation process, providing insight into the model's confidence for each word choice.

## Installation

To use TinyLLaVA Visualizer, simply ensure that you have an environment capable of running TinyLLaVA. If you already have TinyLLaVA set up, you're good to go! No additional installation steps are required for this tool.

```
conda create -n tinyllava_factory python=3.10 -y
conda activate tinyllava_factory
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

#### Usage

### Inference and Visualization

Place the `tinyllava_visualizer.py` in the root directory of the project where your model's code resides. During the model's inference process, integrate the `Monitor` class from the visualizer to generate visual outputs. Below is an example use case:

```
from tinyllava.eval.run_tiny_llava import eval_model
from tinyllava.model.convert_legecy_weights_to_tinyllavafactory import convert_legecy_weights_to_tinyllavafactory
from tinyllava_visualizer import Monitor

def main():
    model = convert_legecy_weights_to_tinyllavafactory('TinyLLaVA-3.1B')
    prompt = "What are the things I should be cautious about when I visit here?"
    image_file = "image_test/1.jpeg"

    args = type('Args', (), {
        "model_path": None,
        "model": model,
        "query": prompt,
        "conv_mode": "phi",  # Adjust based on the LLM version
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    monitor = Monitor(31, args)
    eval_model(args)
    monitor.get_output()

if __name__ == "__main__":
    main()
```

This example demonstrates how to set up and use TinyLLaVA Visualizer in a typical inference workflow. After running this code, the visual outputs will be stored in the `results` directory, categorized by the type of visualization. 

## Project Structure

- `tinyllava_visualizer.py`: The main script for visualization.
- `tinyllava/`: Directory containing core model and data processing code.
- `scripts/`: Contains utility scripts.
- `eval/`: Evaluation scripts and tools.

### Example Visualizations

Here are examples of the types of visual outputs you can expect:

prompt = "What is it?"

image:

<img src="file:///C:/Users/34701/Desktop/2.webp" title="" alt="" width="281">

output:

The image features a small, fluffy, light brown dog with a pink collar. The dog is wearing a **<mark>sweater</mark>**, which adds a touch of warmth and style to its appearance. The dog is standing on a wooden floor, and its gaze is directed straight at the camera, creating a sense of connection between the viewer and the subject. The dog's fur appears soft and fluffy, and its pink collar stands out against its light brown coat. The wooden floor provides a natural and warm background that contrasts with the dog's vibrant colors. The dog's position and the way it looks at the camera give the image a sense of liveliness and personality. The image does not contain any text or other objects. The focus is solely on the dog, making it the central element of the image. The relative position of the dog to the camera and the wooden floor suggests that the photo was taken in a home setting, possibly in the living room or a similar area. The image does not provide any additional context or information about the dog's breed, age, or any other details beyond what can be seen in the image.

### *the visualization of word 'sweater'*

| Word Probability Distributions                                                           | Word Relationships                                                                                           | Word-Image Relationships                                                                |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| <img title="" src="file:///C:/Users/34701/Desktop/word_image_22.png" alt="" width="186"> | <img title="" src="file:///C:/Users/34701/Desktop/TinyLLaVABench-main/word_image_22.png" alt="" width="232"> | <img title="" src="file:///C:/Users/34701/Desktop/hot_image_22.png" alt="" width="250"> |

---

If you encounter any issues or have suggestions, reach out to us at [21376195@buaa.edu.cn].
