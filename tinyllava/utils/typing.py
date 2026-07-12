import logging
from typing import TYPE_CHECKING, overload

import transformers


if TYPE_CHECKING:

    class _Logger(logging.Logger):
        r"""A logger that supports rank0 logging."""
        def info_rank0(self, *args, **kwargs) -> None: ...
        def warning_rank0(self, *args, **kwargs) -> None: ...
        def warning_rank0_once(self, *args, **kwargs) -> None: ...

    class _PreTrainedTokenizer(transformers.PreTrainedTokenizer):
        bos_token_id: int | None
        eos_token_id: int | None
        pad_token_id: int | None
        bos_token: str | None
        eos_token: str | None
        pad_token: str | None

        @overload
        def convert_tokens_to_ids(self, tokens: str) -> int: ...
        @overload
        def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

        def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]: ...

    Logger = _Logger
    PreTrainedTokenizer = _PreTrainedTokenizer

else:
    Logger = logging.Logger
    PreTrainedTokenizer = transformers.PreTrainedTokenizer
