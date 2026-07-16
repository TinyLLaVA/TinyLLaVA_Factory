from __future__ import annotations

from transformers import Trainer

from tinyllava.data.dataset import make_supervised_data_module
from tinyllava.data.processor.creation import create_tinyllava_processor
from tinyllava.utils.model_loading import (
    load_model_config,
    load_training_model,
    load_tokenizer,
    resolve_component_paths,
)
from tinyllava.model.vision_tower.registry import load_image_processor
from tinyllava.train.strategy import get_training_strategy
from tinyllava.utils.config import parse_train_config
from tinyllava.utils.logging import logger_setting, log_trainable_params
from tinyllava.utils.precision import resolve_training_precision


def train():
    model_args, data_args, training_args = parse_train_config()
    logger_setting(getattr(training_args, "output_dir", None))

    paths = resolve_component_paths(model_args)
    model_config = load_model_config(model_args, paths)
    resolve_training_precision(training_args, model_config)
    training_strategy = get_training_strategy(training_args.training_strategy)(
        training_args
    )
    language_model_loading_kwargs = training_strategy.language_model_loading_kwargs()
    model = load_training_model(
        model_args,
        paths,
        model_config,
        language_model_loading_kwargs=language_model_loading_kwargs,
    )
    tokenizer = load_tokenizer(model_args, paths)
    model.tokenizer = tokenizer

    image_processor = load_image_processor(
        paths.image_processor,
        model_type=model.config.vision_config.model_type,
    )
    processor = create_tinyllava_processor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
    )

    # Processor creation can resize embeddings and the output head. Apply the
    # tuning policy after all model mutations are done.
    model = training_strategy(model)
    original_use_cache = getattr(model.config, "use_cache", None)
    if training_args.gradient_checkpointing and original_use_cache is not None:
        model.config.use_cache = False

    # Only the local main process builds a missing Arrow cache. Other workers
    # wait here and then open the completed cache instead of contending for the
    # same datasets file lock.
    with training_args.main_process_first(desc="build the training dataset cache"):
        data_module = make_supervised_data_module(
            processor=processor,
            data_args=data_args,
        )

    log_trainable_params(model)
    trainer = Trainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )

    try:
        trainer.train()
    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
    training_strategy.save(model, trainer)


if __name__ == "__main__":
    train()
