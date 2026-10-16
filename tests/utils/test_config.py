from pathlib import Path
from unittest.mock import patch

import pytest

from tinyllava.train.workflow import training_stages
from tinyllava.train.train import train
from tinyllava.hparams.training import build_training_arguments as build_arguments
from tinyllava.utils.config import load_config

ROOT = Path(__file__).parents[2]
FINETUNE_CONFIGS = [
    ROOT / f"configs/experiments/{name}.yaml"
    for name in ("phi", "phi_lora", "phi_qlora")
]


def test_worker_accepts_deepspeed_launcher_arguments(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "training:\n  saving:\n    output_dir: output/test\n", encoding="utf-8"
    )

    with patch("tinyllava.train.algorithms.sft.run") as run:
        train(
            [
                "--config",
                str(config_path),
                "--deepspeed",
                "configs/deepspeed/zero3.json",
                "--local_rank=2",
            ]
        )

    config = run.call_args.args[0]
    assert config["training"]["runtime"]["deepspeed"] == "configs/deepspeed/zero3.json"
    assert config["training"]["runtime"]["local_rank"] == 2


def test_launcher_arguments_override_yaml_and_dotlist_values(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "training:\n"
        "  saving:\n    output_dir: output/test\n"
        "  runtime:\n    deepspeed: yaml.json\n"
        "    local_rank: 0\n",
        encoding="utf-8",
    )

    with patch("tinyllava.train.algorithms.sft.run") as run:
        train(
            [
                "--config",
                str(config_path),
                "training.runtime.deepspeed=dotlist.json",
                "--deepspeed",
                "cli.json",
                "--local-rank",
                "3",
            ]
        )

    config = run.call_args.args[0]
    assert config["training"]["runtime"]["deepspeed"] == "cli.json"
    assert config["training"]["runtime"]["local_rank"] == 3


def test_precision_is_resolved_before_deepspeed_auto_values(tmp_path):
    deepspeed_path = tmp_path / "zero2.json"
    deepspeed_path.write_text(
        '{"bf16":{"enabled":"auto"},"fp16":{"enabled":"auto"},'
        '"zero_optimization":{"stage":2}}',
        encoding="utf-8",
    )

    arguments = build_arguments(
        {
            "training": {
                "saving": {"output_dir": str(tmp_path / "output")},
                "runtime": {
                    "precision": "bf16",
                    "deepspeed": str(deepspeed_path),
                    "use_cpu": True,
                },
            }
        }
    )

    config = arguments.training.hf_deepspeed_config.config
    assert config["bf16"]["enabled"] is True
    assert config["fp16"]["enabled"] is False


@pytest.mark.parametrize(
    "config_path",
    FINETUNE_CONFIGS,
    ids=lambda path: path.stem,
)
def test_finetune_configs_load_composite_pretraining_checkpoints(config_path):
    config = training_stages(config_path)[-1][1]
    model = config["model"]

    assert model["pretrained_model_name_or_path"]
    assert "language_model_name_or_path" not in model
    assert "vision_model_name_or_path" not in model
    assert "connector" not in model


def test_share_second_pretraining_stage_loads_base_checkpoint():
    stages = training_stages(ROOT / "configs/experiments/phi_share.yaml")
    assert (
        stages[1][1]["model"]["pretrained_model_name_or_path"]
        == stages[0][1]["training"]["saving"]["output_dir"]
    )
    assert "language_model_name_or_path" not in stages[1][1]["model"]


def test_parse_data_source_options_from_yaml(tmp_path):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(
        "data:\n"
        "  dataset_name_or_path: parquet\n"
        "  data_files:\n"
        "    train: [part-1.parquet, part-2.parquet]\n"
        "  split: train[:10%]\n"
        "  cache_dir: /tmp/data-cache\n"
        "training:\n"
        "  saving:\n    output_dir: /tmp/train-output\n"
        "  runtime:\n    use_cpu: true\n"
        "  logging:\n    report_to: []\n"
    )
    arguments = build_arguments(load_config(config_path))
    assert arguments.data.dataset_name_or_path == "parquet"
    assert arguments.data.data_files == {"train": ["part-1.parquet", "part-2.parquet"]}
    assert arguments.data.split == "train[:10%]"
    assert arguments.data.cache_dir == "/tmp/data-cache"
