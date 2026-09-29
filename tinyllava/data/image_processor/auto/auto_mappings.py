"""Lazy local image processors: namespaced package key -> saved image_processor_type.

For example, {"custom__tlf_image_processor": "CustomImageProcessor"} resolves
`tinyllava.data.image_processor.custom.CustomImageProcessor` on demand.
Native HF processors do not need entries here.
"""

from collections import OrderedDict

CUSTOM_IMAGE_PROCESSOR_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict()
