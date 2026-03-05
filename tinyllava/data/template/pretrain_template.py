from dataclasses import dataclass, field
import copy

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from ...utils.constants import IGNORE_INDEX
from . import register_template



@register_template("pretrain")
@dataclass
class PretrainTemplate(Template):
    format_image_token: Formatter = field(default_factory=lambda: EmptyFormatter(slot=""))
    format_user: Formatter = field(default_factory=lambda: EmptyFormatter(slot="<image>"))
    format_assistant: Formatter = field(default_factory=lambda: StringFormatter(slot="{{content}}\n"))
    system: Formatter = field(default_factory=lambda: EmptyFormatter(slot=""))
    separator: Formatter = field(default_factory=lambda: EmptyFormatter(slot=["", ""]))

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        mask_len = len(self.tokenizer_image_token("<image>", tokenizer))
        labels[:mask_len] = IGNORE_INDEX
        return labels
