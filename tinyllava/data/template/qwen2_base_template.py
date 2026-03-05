from dataclasses import dataclass, field

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template


system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."


@register_template("qwen2_base")
@dataclass
class Qwen2BaseTemplate(Template):
    format_image_token: Formatter = field(default_factory=lambda: StringFormatter(slot="<image>\n{{content}}"))
    format_user: Formatter = field(default_factory=lambda: StringFormatter(slot="USER" + ": " + "{{content}}" + " "))
    format_assistant: Formatter = field(default_factory=lambda: StringFormatter(
        slot="ASSISTANT" + ": " + "{{content}}" + "<|endoftext|>"
    ))
    system: Formatter = field(default_factory=lambda: EmptyFormatter(slot=system + " "))
    separator: Formatter = field(default_factory=lambda: EmptyFormatter(slot=[" ASSISTANT: ", "<|endoftext|>"]))
