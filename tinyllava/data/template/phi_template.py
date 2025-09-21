from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template

from transformers import PreTrainedTokenizer
import torch

system = ("A chat between a curious user and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the user's questions.")


@register_template('phi')
@dataclass
class PhiTemplate(Template):
    format_image_token: "Formatter" = StringFormatter.add(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter.add(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter.add(slot="ASSISTANT" + ": " + "{{content}}" + "<|endoftext|>")
    system: "Formatter" = EmptyFormatter.add(slot=system + " ")
    separator: "Formatter" = EmptyFormatter.add(slot=[' ASSISTANT: ', '<|endoftext|>'])
