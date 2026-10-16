"""Supervised training algorithm. Heavy dependencies are loaded only when run."""

from tinyllava.hparams.training import (
    SECTIONS as SECTIONS,
    validate_training_config as validate_config,
    prepare_training_config as prepare_config,
)


__all__ = ["SECTIONS", "validate_config", "prepare_config", "run"]


def run(resolved_config: dict) -> None:
    from tinyllava.data.chat_template.loading import resolve_chat_template
    from tinyllava.data.dataset import make_supervised_data_module
    from tinyllava.data.image_processor import AutoImageProcessor
    from tinyllava.data.processor import TinyLlavaProcessor
    from tinyllava.train.algorithms.sft_trainer import TinyLlavaTrainer
    from tinyllava.train.strategy import get_training_strategy
    from tinyllava.utils.checkpoint import resolve_resume_checkpoint
    from tinyllava.hparams.training import build_training_arguments
    from tinyllava.utils.deepspeed import (
        configure_zero3_gradient_checkpointing,
    )
    from tinyllava.utils.logging import log_trainable_params, logger_setting
    from tinyllava.utils.model_loading import (
        load_model_config,
        load_tokenizer,
        load_training_model,
        resolve_component_paths,
    )
    from tinyllava.utils.precision import (
        cast_model_to_training_dtype,
        resolve_training_precision,
    )

    arguments = build_training_arguments(resolved_config)
    model_args, data_args, training_args = (
        arguments.model,
        arguments.data,
        arguments.training,
    )
    logger_setting(getattr(training_args, "output_dir", None))
    configure_zero3_gradient_checkpointing(training_args)

    paths = resolve_component_paths(model_args)
    model_config = load_model_config(model_args, paths)
    resolve_training_precision(training_args, model_config)
    training_strategy = get_training_strategy(arguments.tuning.strategy)(
        training_args, arguments.tuning
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

    image_processor = AutoImageProcessor.from_pretrained(
        paths.image_processor,
        cache_dir=model_args.cache_dir,
    )
    processor = TinyLlavaProcessor.from_model(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
        chat_template=resolve_chat_template(
            chat_template=model_args.chat_template,
            chat_template_path=model_args.chat_template_path,
        ),
    )

    # Processor creation can resize embeddings and the output head. Apply the
    # tuning policy and dtype normalization after all model mutations are done.
    model = training_strategy(model)
    model = cast_model_to_training_dtype(model, training_args)
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
    trainer = TinyLlavaTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )
    resume_checkpoint = resolve_resume_checkpoint(training_args)

    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
    training_strategy.save(model, trainer)
