from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from tinyllava.hparams.parser import parse_eval_arguments
from tinyllava.train.train import train


@pytest.mark.parametrize(
    "worker,config",
    [
        (
            train,
            {
                "model": {"pretrained_model_name_or_path": "output/align"},
                "training": {
                    "saving": {"output_dir": "output/custom"},
                    "logging": {"run_name": "custom"},
                },
            },
        ),
        (
            parse_eval_arguments,
            {
                "model": {"model_name_or_path": "output/finetune"},
                "output": {"answers_file": "custom/answers.jsonl"},
            },
        ),
    ],
)
def test_worker_config_is_not_recomposed(tmp_path, worker, config):
    path = tmp_path / "resolved.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    target = (
        "tinyllava.train.algorithms.sft.run"
        if worker is train
        else "tinyllava.hparams.parser.build_eval_arguments"
    )
    with patch(target) as consume:
        worker(["--config", str(path)])
    consume.assert_called_once_with(config)
