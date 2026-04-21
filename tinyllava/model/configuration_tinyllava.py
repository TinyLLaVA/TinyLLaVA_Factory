from transformers import PretrainedConfig
from transformers import CONFIG_MAPPING
from transformers import AutoConfig
from tinyllava.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


# TODO: (incompatible change) replace with `PreTrainedConfig` as hf transformers
# corrected CamelCasing in https://github.com/huggingface/transformers/pull/41300
class TinyLlavaConfig(PretrainedConfig):
    model_type = "tinyllava"

    # Model path and tokenizer configuration
    llm_model_name_or_path: str = ""
    tokenizer_name_or_path: str | None = None
    vision_model_name_or_path: str = ""
    vision_model_name_or_path2: str = ""
    connector_type: str | None = None

    # Sub-configs
    text_config: dict | PretrainedConfig | None = None
    vision_config: dict | PretrainedConfig | None = None

    # Tokenizer parameters
    pad_token: str | None = None
    pad_token_id: int | None = None
    tokenizer_padding_side: str = "right"
    tokenizer_model_max_length: int = 2048
    tokenizer_use_fast: bool = False

    # Vision tower parameters
    vision_feature_layer: int = -2
    vision_feature_select_strategy: str = "patch"
    image_aspect_ratio: str = "square"

    # Connector/projector parameters
    resampler_hidden_size: int | None = None
    num_queries: int | None = None
    num_resampler_layers: int | None = None

    # Training parameters
    tune_type_llm: str = "frozen"
    tune_type_connector: str = "frozen"
    tune_type_vision_tower: str = "frozen"
    tune_vision_tower_from_layer: int = -1

    # Cache and constants
    use_cache: bool = False
    cache_dir: str | None = None
    ignore_index: int = IGNORE_INDEX
    image_token_index: int = IMAGE_TOKEN_INDEX

    # Derived attributes
    hidden_size: int | None = None
    vocab_size: int | None = None
    vision_hidden_size: int | None = None

    def __post_init__(self, **kwargs):
        if self.tokenizer_name_or_path is None or self.tokenizer_name_or_path == "":
            self.tokenizer_name_or_path = self.llm_model_name_or_path

        self._load_text_config(self.text_config)
        self._load_vision_config(self.vision_config)

        super().__post_init__(**kwargs)

    def load_from_config(self, config):
        self.llm_model_name_or_path = getattr(config, "model_name_or_path", "")
        self.tokenizer_name_or_path = (
            getattr(config, "tokenizer_name_or_path", None)
            or self.llm_model_name_or_path
        )
        self.vision_model_name_or_path = getattr(config, "vision_tower", "")
        self.vision_model_name_or_path2 = getattr(config, "vision_tower2", "")
        self.connector_type = getattr(config, "connector_type", None)
        self.vision_feature_layer = getattr(config, "mm_vision_select_layer", -2)
        self.vision_feature_select_strategy = getattr(
            config, "mm_vision_select_feature", "patch"
        )
        self.image_aspect_ratio = getattr(config, "image_aspect_ratio", "pad")
        self.resampler_hidden_size = getattr(config, "resampler_hidden_size", None)
        self.num_queries = getattr(config, "num_queries", None)
        self.num_resampler_layers = getattr(config, "num_resampler_layers", None)

        self.cache_dir = getattr(config, "cache_dir", None)
        self.tokenizer_use_fast = getattr(config, "tokenizer_use_fast", False)
        self.tokenizer_model_max_length = getattr(config, "model_max_length", 2048)
        self.tokenizer_padding_side = getattr(config, "tokenizer_padding_side", "right")

        self._load_text_config()
        self._load_vision_config()

    def _load_text_config(self, text_config=None):
        if self.llm_model_name_or_path is None or self.llm_model_name_or_path == "":
            self.text_config = CONFIG_MAPPING["llama"]()

        else:
            self.text_config = AutoConfig.from_pretrained(
                self.llm_model_name_or_path, trust_remote_code=True
            )
            if text_config is not None:
                self.text_config = self.text_config.from_dict(text_config)

        self.hidden_size = getattr(
            self.text_config,
            "hidden_size",
            getattr(self.text_config, "model_dim", None),
        )
        self.vocab_size = getattr(self.text_config, "vocab_size", None)

    def _load_vision_config(self, vision_config=None):
        if (
            self.vision_model_name_or_path is None
            or self.vision_model_name_or_path == ""
        ):
            self.vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        else:
            self.vision_config = AutoConfig.from_pretrained(
                self.vision_model_name_or_path.split(":")[-1]
            )
            self.vision_config = getattr(
                self.vision_config, "vision_config", self.vision_config
            )
            if vision_config is not None:
                self.vision_config = self.vision_config.from_dict(vision_config)

        self.vision_config.model_name_or_path = self.vision_model_name_or_path.split(
            ":"
        )[-1]
        self.vision_config.model_name_or_path2 = self.vision_model_name_or_path2.split(
            ":"
        )[-1]
        self.vision_hidden_size = getattr(self.vision_config, "hidden_size", None)
