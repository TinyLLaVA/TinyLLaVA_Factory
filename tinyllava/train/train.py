"""Training worker: read resolved YAML and execute the selected algorithm."""

import argparse
from collections.abc import Sequence

from tinyllava.train.algorithms import get_algorithm
from tinyllava.utils.config import load_config, merge_mappings


def train(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved training YAML.")
    parser.add_argument("--deepspeed", help="Override training.deepspeed.")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    options = parser.parse_args(args)

    config = load_config(options.config, options.overrides)
    runtime = {
        key: value
        for key, value in {
            "deepspeed": options.deepspeed,
            "local_rank": options.local_rank,
        }.items()
        if value is not None
    }
    if runtime:
        config = merge_mappings(config, {"training": runtime})
    get_algorithm(config.get("algorithm", "sft")).run(config)


if __name__ == "__main__":
    train()
