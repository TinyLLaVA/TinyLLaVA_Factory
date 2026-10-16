"""Type annotations for the rank-aware logging methods installed by logging.py."""

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:

    class Logger(logging.Logger):
        def info_rank0(self, *args, **kwargs) -> None: ...
        def debug_rank0(self, *args, **kwargs) -> None: ...
        def warning_rank0(self, *args, **kwargs) -> None: ...

else:
    Logger = logging.Logger
