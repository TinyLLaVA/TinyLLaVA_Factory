"""Project multimodal processors indexed by composite model configuration."""

from collections import OrderedDict

PROCESSOR_CONFIG_MAPPING_NAMES = OrderedDict(tinyllava="TinyLlavaConfig")
PROCESSOR_MAPPING_NAMES = OrderedDict(tinyllava="TinyLlavaProcessor")

# Paths are independent of model_type and processor package naming.
PROCESSOR_CONFIG_MODULE_NAMES = OrderedDict(
    tinyllava="tinyllava.model.configuration_tinyllava",
)
PROCESSOR_MODULE_NAMES = OrderedDict(
    tinyllava="tinyllava.data.processor.tinyllava",
)
