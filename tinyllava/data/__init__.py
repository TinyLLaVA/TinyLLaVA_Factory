from typing import TYPE_CHECKING

from transformers.utils.import_utils import (
    _LazyModule,
    define_import_structure,
)


if TYPE_CHECKING:
    from .assistant_mask import *
    from .chat_template import *
    from .collator import *
    from .dataset import *
    from .image_payload import *
    from .image_processor import *
    from .message_format import *
    from .processor import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
