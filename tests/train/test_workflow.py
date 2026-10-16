import sys
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from tinyllava.train.workflow import run

ROOT = Path(__file__).parents[2]


def test_distributed_launch_uses_selected_devices_and_resolved_batch_size():
    def check_command(command, *, cwd, env, check):
        assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
        assert "--nproc_per_node=2" in command
        assert "--master_port=29999" in command
        assert command[command.index("--module") + 1] == "tinyllava.train.train"
        assert env["CUDA_VISIBLE_DEVICES"] == "2,5"
        assert check is True
        config = OmegaConf.load(command[-1])
        assert config.training.gradient_accumulation_steps == 4
        assert config.model.language_model_name_or_path

    with patch("tinyllava.runner.subprocess.run", side_effect=check_command) as launch:
        run(
            ROOT / "configs/experiments/qwen2_instruct.yaml",
            [
                "launcher.distributed=true",
                "launcher.devices=[2,5]",
                "launcher.master_port=29999",
                "stages.pretrain.training.global_batch_size=16",
                "stages.pretrain.training.per_device_train_batch_size=2",
            ],
            steps=["pretrain"],
        )
    launch.assert_called_once()
    assert not Path(launch.call_args.args[0][-1]).exists()
