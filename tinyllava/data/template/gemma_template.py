from dataclasses import dataclass, field

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template
from ...utils.constants import IGNORE_INDEX


system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."


@register_template("gemma")
@dataclass
class GemmaTemplate(Template):
    format_image_token: Formatter = field(default_factory=lambda: StringFormatter(slot="<image>\n{{content}}"))
    format_user: Formatter = field(default_factory=lambda: StringFormatter(slot="USER" + ": " + "{{content}}" + " "))
    format_assistant: Formatter = field(default_factory=lambda: StringFormatter(
        slot="ASSISTANT" + ": " + "{{content}}" + "<eos>"
    ))
    system: Formatter = field(default_factory=lambda: EmptyFormatter(slot=system + " "))
    separator: Formatter = field(default_factory=lambda: EmptyFormatter(slot=[" ASSISTANT: ", "<eos>"]))

    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 1  # bos
        eos_token_length = 1
        bos_token_length = 1
        labels[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = (
                len(self.tokenizer_image_token(rou, tokenizer))
                + eos_token_length
                - bos_token_length
            )
            instruction_len = (
                len(self.tokenizer_image_token(parts[0], tokenizer))
                - 1
                - bos_token_length
            )
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len
