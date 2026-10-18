"""Overfit two colors through the real training worker and checkpoint loader."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    LlamaConfig,
    PreTrainedTokenizerFast,
    Qwen2Config,
    SiglipImageProcessor,
    SiglipVisionConfig,
    set_seed,
)

from tinyllava.cli import main
from tinyllava.data.processor import TinyLlavaProcessor
from tinyllava.eval.generation import generate_response, make_user_message
from tinyllava.model import TinyLlavaConfig, TinyLlavaForConditionalGeneration
from tinyllava.model.llm import AutoLanguageModelForCausalLM
from tinyllava.utils.model_loading import load_tinyllava_checkpoint_bundle

PRESETS = sorted((Path(__file__).parents[2] / "configs/training").glob("*.yaml"))
QUESTION = "What color is this?"
# A single-token answer directly follows the image. This lets connector-only
# training learn the task even when the language model is randomly initialized.
CHAT_TEMPLATE = """\
{% for message in messages %}
{% if message['role'] == 'user' %}
{% for item in message['content'] if item['type'] == 'text' %}{{ item['text'] + ' ' }}{% endfor %}
{% for item in message['content'] if item['type'] == 'image' %}{{ '<image> ' }}{% endfor %}
{% else %}
{% generation %}{% for item in message['content'] %}{{ item['text'] }}{% endfor %}{% endgeneration %}
{% endif %}
{% endfor %}
"""


def make_experiment(
    tmp_path, monkeypatch, preset, seed, *, qwen_siglip=False, components=False
):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)
    set_seed(seed)

    words = [
        "[UNK]",
        "[PAD]",
        "[EOS]",
        "<image>",
        "What",
        "color",
        "is",
        "this",
        "?",
        "red",
        "blue",
    ]
    backend = Tokenizer(
        WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        additional_special_tokens=["<image>"],
        chat_template=CHAT_TEMPLATE,
    )
    model = TinyLlavaForConditionalGeneration(
        TinyLlavaConfig(
            text_config=(Qwen2Config if qwen_siglip else LlamaConfig)(
                vocab_size=len(tokenizer),
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=None,
            ),
            vision_config=(SiglipVisionConfig if qwen_siglip else CLIPVisionConfig)(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                image_size=16,
                patch_size=8,
            ),
        )
    )
    processor = TinyLlavaProcessor.from_model(
        model=model,
        tokenizer=tokenizer,
        image_processor=SiglipImageProcessor(size={"height": 16, "width": 16})
        if qwen_siglip
        else CLIPImageProcessor(
            size={"shortest_edge": 16},
            crop_size={"height": 16, "width": 16},
        ),
    )
    base, output = tmp_path / "base", tmp_path / "output"
    model.save_pretrained(base)
    processor.save_pretrained(base)
    if components:
        AutoLanguageModelForCausalLM.from_config(
            model.config.text_config
        ).save_pretrained(tmp_path / "language")
        tokenizer.save_pretrained(tmp_path / "language")
        model.model.vision_tower.save_pretrained(tmp_path / "vision")
        processor.image_processor.save_pretrained(tmp_path / "vision")

    records = []
    for index, color in enumerate(("red", "blue")):
        Image.new("RGB", (16, 16), color).save(tmp_path / f"{index}.png")
        records.append(
            {
                "image": f"{index}.png",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{QUESTION}"},
                    {"from": "gpt", "value": color},
                ],
            }
        )
    annotations = tmp_path / "colors.json"
    annotations.write_text(json.dumps(records))
    config = {
        "algorithm": "sft",
        "model": {
            "pretrained_model_name_or_path": str(base),
            "tokenizer_use_fast": True,
            "model_max_length": 64,
        },
        "data": {
            "dataset_name_or_path": str(annotations),
            "image_folder": str(tmp_path),
            "cache_dir": str(tmp_path / "cache"),
        },
        "training": {
            "from": str(preset),
            "max_steps": 300,
            "learning_rate": 0.001,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "runtime": {
                "use_cpu": True,
                "precision": "fp32",
                "seed": seed,
            },
            "dataloader": {"dataloader_num_workers": 0, "dataloader_pin_memory": False},
            "saving": {"output_dir": str(output), "save_strategy": "no"},
            "logging": {
                "report_to": [],
                "disable_tqdm": True,
                "logging_strategy": "no",
            },
            "scheduler": {"lr_scheduler_type": "constant", "warmup_steps": 0},
        },
    }
    if components:
        config["model"].pop("pretrained_model_name_or_path")
        config["model"].update(
            language_model_name_or_path=str(tmp_path / "language"),
            vision_model_name_or_path=str(tmp_path / "vision"),
        )
    return config


@pytest.mark.slow
@pytest.mark.parametrize("preset", PRESETS, ids=lambda path: path.stem)
@pytest.mark.parametrize("seed", [0, 42])
def test_train_and_generate_colors(tmp_path, monkeypatch, preset, seed):
    config = make_experiment(tmp_path, monkeypatch, preset, seed)
    output = tmp_path / "output"
    path = tmp_path / "experiment.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    main(["train", "--config", str(path)])

    assert json.loads((output / "trainer_state.json").read_text())["global_step"] == 300
    bundle = load_tinyllava_checkpoint_bundle(str(output), device="cpu")
    bundle.model.eval()
    for color in ("red", "blue"):
        answer = generate_response(
            model=bundle.model,
            processor=bundle.processor,
            messages=[make_user_message(QUESTION, [Image.new("RGB", (16, 16), color)])],
            temperature=0,
            max_new_tokens=1,
        )
        assert answer == color


@pytest.mark.slow
def test_qwen_siglip_stages_and_adapter_resume(tmp_path, monkeypatch):
    presets = Path(__file__).parents[2] / "configs/training"
    config = make_experiment(
        tmp_path,
        monkeypatch,
        presets / "pretrain.yaml",
        42,
        qwen_siglip=True,
        components=True,
    )
    config["training"]["max_steps"] = 2
    config["training"]["saving"] = {"save_strategy": "steps", "save_steps": 1}
    config["output_root"] = str(tmp_path / "runs")
    stage_options = deepcopy(config["training"])
    stage_options.pop("max_steps")
    config["stages"] = {
        "pretrain": {},
        "repeat": {"checkpoint": None},
        "lora": {
            "training": {**stage_options, "from": str(presets / "lora_finetune.yaml")}
        },
        "lora2": {
            "training": {**stage_options, "from": str(presets / "lora_finetune.yaml")}
        },
        "full": {"training": {**stage_options, "from": str(presets / "finetune.yaml")}},
    }
    path = tmp_path / "stages.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    main(["train", "--config", str(path), "--steps", "pretrain,repeat,lora"])
    output = tmp_path / "runs/stages"
    from safetensors.torch import load_file

    first = load_file(output / "pretrain/model.safetensors")
    repeated = load_file(output / "repeat/model.safetensors")
    for name in first:
        torch.testing.assert_close(first[name], repeated[name], rtol=0, atol=0)
    adapter = output / "lora"
    assert (adapter / "checkpoint-2/adapter_model.safetensors").is_file()
    # Resume must restore both adapter/connector weights and the optimizer state.
    main(["train", "--config", str(path), "--steps", "lora", "training.max_steps=3"])
    assert json.loads((adapter / "trainer_state.json").read_text())["global_step"] == 3
    main(["train", "--config", str(path), "--steps", "lora2,full"])
    second_adapter = json.loads((output / "lora2/adapter_config.json").read_text())
    assert second_adapter["base_model_name_or_path"] == str(adapter)
    bundle = load_tinyllava_checkpoint_bundle(str(output / "full"), device="cpu")
    assert bundle.model.config.text_config.model_type == "qwen2"
    assert bundle.model.config.vision_config.model_type == "siglip_vision_model"
    assert (output / "full/model.safetensors").is_file()
