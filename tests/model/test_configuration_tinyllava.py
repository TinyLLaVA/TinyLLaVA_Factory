"""
Regression tests for TinyLlavaConfig compatibility after constructor refactor.

This file ensures:
1. All parameter default values remain unchanged.
2. Initialization call patterns remain compatible (no args, partial args, full args).
3. Parameter combinations still behave correctly.
4. Config loading and mapping behavior is unchanged.
"""
import json
from pathlib import Path

from transformers import CONFIG_MAPPING

from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.utils.arguments import ModelArguments


def _write_config(tmp_path: Path, folder_name: str, content: dict) -> str:
    model_dir = tmp_path / folder_name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(content), encoding="utf-8")
    return str(model_dir)


class TestTinyLlavaConfig:
    def test_init_without_model_paths_uses_local_default_configs(self):
        config = TinyLlavaConfig()

        assert isinstance(config.text_config, type(CONFIG_MAPPING["llama"]()))
        assert config.hidden_size == config.text_config.hidden_size
        assert config.vocab_size == config.text_config.vocab_size

        assert config.vision_config.model_type == "clip_vision_model"
        assert config.vision_config.hidden_size == 1024
        assert config.vision_config.image_size == 336
        assert config.vision_config.model_name_or_path == ""
        assert config.vision_config.model_name_or_path2 == ""
        assert config.vision_hidden_size == 1024

    def test_all_parameter_default_values_preserved(self):
        """Verify all parameter defaults are preserved after refactor."""
        config = TinyLlavaConfig()

        # Model path and tokenizer defaults
        assert config.llm_model_name_or_path == ""
        # tokenizer_name_or_path is normalized to llm_model_name_or_path in __post_init__.
        assert config.tokenizer_name_or_path == ""
        assert config.vision_model_name_or_path == ""
        assert config.vision_model_name_or_path2 == ""
        assert config.connector_type is None

        # Tokenizer defaults
        assert config.pad_token is None
        assert config.pad_token_id is None
        assert config.tokenizer_padding_side == "right"
        assert config.tokenizer_model_max_length == 2048
        assert config.tokenizer_use_fast is False

        # Vision tower defaults
        assert config.vision_feature_layer == -2
        assert config.vision_feature_select_strategy == "patch"
        assert config.image_aspect_ratio == "square"

        # Connector/projector defaults
        assert config.resampler_hidden_size is None
        assert config.num_queries is None
        assert config.num_resampler_layers is None

        # Training defaults
        assert config.tune_type_llm == "frozen"
        assert config.tune_type_connector == "frozen"
        assert config.tune_type_vision_tower == "frozen"
        assert config.tune_vision_tower_from_layer == -1

        # Cache and constants
        assert config.use_cache is False
        assert config.cache_dir is None

    def test_init_with_custom_parameters_overrides_defaults(self, tmp_path: Path):
        """Verify custom parameters correctly override defaults."""
        llm_path = _write_config(
            tmp_path,
            "llm",
            {
                "model_type": "llama",
                "hidden_size": 512,
                "vocab_size": 4096,
            },
        )
        vision_path = _write_config(
            tmp_path,
            "vision",
            {
                "model_type": "clip_vision_model",
                "hidden_size": 768,
                "image_size": 224,
            },
        )

        config = TinyLlavaConfig(
            llm_model_name_or_path=llm_path,
            vision_model_name_or_path=f"clip:{vision_path}",
            vision_model_name_or_path2="dinov2:/tmp/vision2",
            text_config={"hidden_size": 1536, "vocab_size": 50000},
            vision_config={"hidden_size": 1536, "image_size": 512},
        )

        assert config.hidden_size == 1536
        assert config.vocab_size == 50000
        assert config.vision_hidden_size == 1536
        assert config.vision_config.image_size == 512
        assert config.vision_config.model_name_or_path == vision_path
        assert config.vision_config.model_name_or_path2 == "/tmp/vision2"

    def test_partial_parameter_initialization_works(self, tmp_path: Path):
        """Verify partial initialization works and other fields keep defaults."""
        llm_path = _write_config(
            tmp_path,
            "partial_llm",
            {
                "model_type": "llama",
                "hidden_size": 512,
                "vocab_size": 4096,
            },
        )
        vision_path = _write_config(
            tmp_path,
            "partial_vision",
            {
                "model_type": "clip_vision_model",
                "hidden_size": 768,
                "image_size": 224,
            },
        )

        config = TinyLlavaConfig(
            llm_model_name_or_path=llm_path,
            vision_model_name_or_path=f"clip:{vision_path}",
            tune_type_llm="lora",
        )

        assert config.llm_model_name_or_path == llm_path
        assert config.tokenizer_name_or_path == llm_path
        assert config.vision_model_name_or_path == f"clip:{vision_path}"
        assert config.tune_type_llm == "lora"
        # Other parameters should still keep their defaults.
        assert config.tune_type_connector == "frozen"
        assert config.tune_type_vision_tower == "frozen"
        assert config.tokenizer_padding_side == "right"
        assert config.image_aspect_ratio == "square"

    def test_load_from_config_maps_model_arguments_used_by_train(self, tmp_path: Path):
        """Verify load_from_config maps ModelArguments fields correctly."""
        llm_path = _write_config(
            tmp_path,
            "llm",
            {
                "model_type": "llama",
                "hidden_size": 768,
                "vocab_size": 32000,
            },
        )
        vision_path = _write_config(
            tmp_path,
            "vision",
            {
                "model_type": "clip_vision_model",
                "hidden_size": 768,
                "image_size": 336,
            },
        )

        model_args = ModelArguments(
            model_name_or_path=llm_path,
            tokenizer_name_or_path=None,
            vision_tower=f"clip:{vision_path}",
            vision_tower2="dinov2:/tmp/vision2",
            connector_type="mlp2x_gelu",
            mm_vision_select_layer=-4,
            mm_vision_select_feature="cls_patch",
            model_max_length=512,
            tokenizer_padding_side="left",
            tokenizer_use_fast=True,
            cache_dir="/tmp/cache",
            resampler_hidden_size=256,
            num_queries=64,
            num_resampler_layers=2,
        )

        config = TinyLlavaConfig()
        config.load_from_config(model_args)

        assert config.llm_model_name_or_path == llm_path
        assert config.tokenizer_name_or_path == llm_path
        assert config.vision_model_name_or_path == f"clip:{vision_path}"
        assert config.vision_model_name_or_path2 == "dinov2:/tmp/vision2"
        assert config.connector_type == "mlp2x_gelu"
        assert config.vision_feature_layer == -4
        assert config.vision_feature_select_strategy == "cls_patch"
        assert config.image_aspect_ratio == "pad"
        assert config.tokenizer_model_max_length == 512
        assert config.tokenizer_padding_side == "left"
        assert config.tokenizer_use_fast is True
        assert config.cache_dir == "/tmp/cache"
        assert config.resampler_hidden_size == 256
        assert config.num_queries == 64
        assert config.num_resampler_layers == 2
        assert config.hidden_size == 768
        assert config.vision_hidden_size == 768
