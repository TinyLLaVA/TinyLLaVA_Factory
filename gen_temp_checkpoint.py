from pathlib import Path

from tinyllava.model import *

config_path = Path('~/.cache/huggingface/hub/').expanduser()
config_path = config_path.joinpath('models--Zhang199--TinyLLaVA-Qwen2-0.5B-SigLIP')
config_path = config_path.joinpath('snapshots/6aef66ed2e0125f57a5ec562fe3c0bf1204d8fa3/config.json')

model_config = TinyLlavaConfig.from_json_file(config_path)
print(model_config)
