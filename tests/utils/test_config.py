from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from tinyllava.utils.config import build_train_arguments, parse_train_config


ROOT = Path(__file__).parents[2]
FINETUNE_CONFIGS = sorted((ROOT / "configs" / "train").glob("*finetune.yaml"))
FINETUNE_CONFIGS += sorted(
    (ROOT / "configs" / "train" / "models").glob("*finetune.yaml")
)


def test_parse_train_config_accepts_deepspeed_launcher_arguments(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text("training:\n  output_dir: output/test\n", encoding="utf-8")

    with patch(
        "tinyllava.utils.config.build_train_arguments",
        side_effect=lambda config: config,
    ):
        config = parse_train_config(
            [
                "--config",
                str(config_path),
                "--deepspeed",
                "scripts/zero3.json",
                "--local_rank=2",
            ]
        )

    assert config["training"]["deepspeed"] == "scripts/zero3.json"
    assert config["training"]["local_rank"] == 2


def test_launcher_arguments_override_yaml_and_dotlist_values(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "training:\n"
        "  output_dir: output/test\n"
        "  deepspeed: yaml.json\n"
        "  local_rank: 0\n",
        encoding="utf-8",
    )

    with patch(
        "tinyllava.utils.config.build_train_arguments",
        side_effect=lambda config: config,
    ):
        config = parse_train_config(
            [
                "--config",
                str(config_path),
                "training.deepspeed=dotlist.json",
                "--deepspeed",
                "cli.json",
                "--local-rank",
                "3",
            ]
        )

    assert config["training"]["deepspeed"] == "cli.json"
    assert config["training"]["local_rank"] == 3


def test_precision_is_resolved_before_deepspeed_auto_values(tmp_path):
    deepspeed_path = tmp_path / "zero2.json"
    deepspeed_path.write_text(
        '{"bf16":{"enabled":"auto"},"fp16":{"enabled":"auto"},'
        '"zero_optimization":{"stage":2}}',
        encoding="utf-8",
    )

    _, _, training_args = build_train_arguments(
        {
            "training": {
                "output_dir": str(tmp_path / "output"),
                "precision": "bf16",
                "deepspeed": str(deepspeed_path),
                "use_cpu": True,
            }
        }
    )

    config = training_args.hf_deepspeed_config.config
    assert config["bf16"]["enabled"] is True
    assert config["fp16"]["enabled"] is False


@pytest.mark.parametrize(
    "config_path",
    FINETUNE_CONFIGS,
    ids=lambda path: path.stem,
)
def test_finetune_configs_load_composite_pretraining_checkpoints(config_path):
    config = OmegaConf.to_container(
        OmegaConf.load(config_path),
        resolve=False,
    )
    model = config["model"]

    assert model["pretrained_model_name_or_path"]
    assert "language_model_name_or_path" not in model
    assert "vision_model_name_or_path" not in model
    assert "connector_config" not in model


def test_share_second_pretraining_stage_loads_base_checkpoint():
    config = OmegaConf.to_container(
        OmegaConf.load(
            ROOT / "configs" / "train" / "models" / "phi_share_pretrain.yaml"
        ),
        resolve=False,
    )

    assert config["model"]["pretrained_model_name_or_path"] == (
        "output/tinyllava-phi-share-base-pretrain"
    )
