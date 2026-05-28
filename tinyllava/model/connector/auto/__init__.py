from typing import TYPE_CHECKING

from transformers.utils.import_utils import (
    _LazyModule,
    define_import_structure,
)


if TYPE_CHECKING:
    from .configuration_auto import *
    from .modeling_auto import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
