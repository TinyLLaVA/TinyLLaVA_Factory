from tinyllava.eval.run_tiny_llava import eval_model
from tinyllava.model.convert_legecy_weights_to_tinyllavafactory import *
from tinyllava_visualizer import *

model = convert_legecy_weights_to_tinyllavafactory('TinyLLaVA-3.1B')
prompt = "What is it?"
image_file = "image_test/2.webp"
args = type('Args', (), {
    "model_path": None,
    "model": model,
    "query": prompt,
    "conv_mode": "phi",
    # the same as conv_version in the training stage. Different LLMs have different conv_mode/conv_version, please replace it
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

monitor = Monitor(args, llm_layers_index=31)
eval_model(args)
monitor.get_output(output_dir='results/')
