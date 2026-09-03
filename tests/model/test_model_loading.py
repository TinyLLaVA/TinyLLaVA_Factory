from unittest.mock import patch

from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.utils.arguments import ModelArguments
from tinyllava.utils.model_loading import (
    ComponentPaths,
    load_model_config,
    load_training_model,
    resolve_component_paths,
)


def test_complete_checkpoint_takes_precedence_over_component_defaults(tmp_path):
    config = TinyLlavaConfig()
    config.save_pretrained(tmp_path)
    model_args = ModelArguments(
        pretrained_model_name_or_path=str(tmp_path),
    )

    paths = resolve_component_paths(model_args)
    loaded_config = load_model_config(model_args, paths)

    assert paths.pretrained_model == str(tmp_path)
    assert paths.language_model == str(tmp_path)
    assert paths.vision_model == str(tmp_path)
    assert paths.tokenizer == str(tmp_path)
    assert paths.image_processor == str(tmp_path)
    assert isinstance(loaded_config, TinyLlavaConfig)


def test_component_paths_use_explicit_language_model_name():
    model_args = ModelArguments(
        language_model_name_or_path="language-model",
        vision_model_name_or_path="vision-model",
    )

    paths = resolve_component_paths(model_args)

    assert paths.language_model == "language-model"
    assert paths.tokenizer == "language-model"


def test_training_model_uses_standard_from_pretrained_for_checkpoint():
    model_args = ModelArguments(
        pretrained_model_name_or_path="complete-checkpoint",
        cache_dir="cache",
        attn_implementation="sdpa",
    )
    paths = ComponentPaths(
        pretrained_model="complete-checkpoint",
        language_model="complete-checkpoint",
        tokenizer="complete-checkpoint",
        vision_model="complete-checkpoint",
        image_processor="complete-checkpoint",
    )
    config = TinyLlavaConfig()
    sentinel = object()

    with patch.object(
        TinyLlavaForConditionalGeneration,
        "from_pretrained",
        return_value=sentinel,
    ) as from_pretrained:
        model = load_training_model(
            model_args,
            paths,
            config,
            language_model_loading_kwargs={"torch_dtype": "dtype"},
        )

    assert model is sentinel
    from_pretrained.assert_called_once_with(
        "complete-checkpoint",
        config=config,
        torch_dtype="dtype",
        cache_dir="cache",
        attn_implementation={"text_config": "sdpa"},
    )


def test_training_model_assembles_components_without_checkpoint():
    model_args = ModelArguments(cache_dir="cache")
    paths = ComponentPaths(
        pretrained_model=None,
        language_model="language-model",
        tokenizer="tokenizer",
        vision_model="vision-model",
        image_processor="vision-model",
    )
    config = TinyLlavaConfig()
    sentinel = object()

    with patch.object(
        TinyLlavaForConditionalGeneration,
        "from_pretrained_components",
        return_value=sentinel,
    ) as from_components:
        model = load_training_model(model_args, paths, config)

    assert model is sentinel
    from_components.assert_called_once_with(
        config,
        language_model_name_or_path="language-model",
        vision_model_name_or_path="vision-model",
        language_model_loading_kwargs={
            "cache_dir": "cache",
            "attn_implementation": None,
        },
        vision_model_loading_kwargs={"cache_dir": "cache"},
    )
