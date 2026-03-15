from dataclasses import dataclass

from .formatter import EmptyFormatter, StringFormatter, Formatter
from .base import Template
from . import register_template
from ...utils.constants import *

system = ("A chat between a curious user and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the user's questions.")


@register_template('gemma')
@dataclass
class GemmaTemplate(Template):
    format_image_token: "Formatter" = StringFormatter.add(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter.add(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter.add(slot="ASSISTANT" + ": " + "{{content}}" + "<eos>")
    system: "Formatter" = EmptyFormatter.add(slot=system + " ")
    separator: "Formatter" = EmptyFormatter.add(slot=[' ASSISTANT: ', '<eos>'])

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
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length - bos_token_length
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1 - bos_token_length
            labels[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len
