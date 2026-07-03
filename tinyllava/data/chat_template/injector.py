"""Inject TinyLLaVA anchors into common Hugging Face chat templates.

This module intentionally does not parse full Jinja syntax. It preserves the
surrounding template while replacing content expressions with a TinyLLaVA-aware
renderer and assistant generation spans.

Typical source shapes:
  {{ message.content }}
  {{ message["content"] }}
  {{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' }}
  {{ ' ' + message["content"]|trim + eos_token }}
  {% set content = message.content %}
"""

import re
from dataclasses import dataclass
from typing import Literal


_GENERATION_RE = re.compile(r"\{\%-?\s*generation\s*-?\%\}")

# Matches a print block that is exactly a content reference:
#   {{ message.content }}
#   {{- message['content'] -}}
# This fast path is safe because the whole expression is just content.
_CONTENT_EXPR_RE = re.compile(
    r"\{\{\s*-?\s*"
    r"(?P<message>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"(?:\s*\.\s*content|\s*\[\s*['\"]content['\"]\s*\])"
    r"\s*-?\s*\}\}"
)

# Matches any Jinja print block and captures the inner expression.  We inspect
# this when content is embedded in a larger expression, for example:
#   {{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' }}
_PRINT_BLOCK_RE = re.compile(r"\{\{\s*-?\s*(?P<expr>.*?)\s*-?\s*\}\}", re.DOTALL)

# Finds a content reference inside a larger expression.  The named "message"
# group lets us guard generation spans with `message.role == 'assistant'`.
_MESSAGE_CONTENT_REF_RE = re.compile(
    r"(?P<message>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"(?:\s*\.\s*content|\s*\[\s*['\"]content['\"]\s*\])"
)

# Used after a template normalizes content into a temporary variable:
#   {% set content = message.content %}
#   {{ content }}
# This must not match dictionary keys or attributes such as:
#   messages[0]['content']
#   message.content
_CONTENT_VARIABLE_RE = re.compile(r"(?<![.\'\"\[])\bcontent\b(?![\'\"\]])")
_SET_CONTENT_FROM_MESSAGE_RE = re.compile(
    r"(?P<set>\{\%-?\s*set\s+content\s*=\s*)"
    r"(?P<message>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"(?P<content_ref>(?:\s*\.\s*content|\s*\[\s*['\"]content['\"]\s*\]))"
    r"(?P<end>\s*-?\%\})"
)

# Fallback for templates that loop directly over message.content instead of
# printing it as one expression:
#   {% for item in message.content %}...{% endfor %}
_CONTENT_LOOP_RE = re.compile(
    r"(?P<loop>"
    r"\{\%-?\s*for\s+[a-zA-Z_][a-zA-Z0-9_]*\s+in\s+"
    r"(?P<message>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"(?:\s*\.\s*content|\s*\[\s*['\"]content['\"]\s*\])"
    r"\s*-?\%\}"
    r".*?"
    r"\{\%-?\s*endfor\s*-?\%\}"
    r")",
    re.DOTALL,
)

ImagePlacement = Literal["images_first", "interleaved"]


@dataclass(frozen=True, slots=True)
class ChatTemplateInjectionResult:
    chat_template: str
    added_content_renderer: bool = False
    added_generation_block: bool = False
    warnings: tuple[str, ...] = ()


def inject_tinyllava_anchors(
    chat_template: str,
    image_token: str = "<image>\n",
    image_placement: ImagePlacement = "images_first",
    special_content_handlers: tuple[str, ...] = (),
) -> ChatTemplateInjectionResult:
    _validate_image_placement(image_placement)
    needs_content_renderer = not _handles_multimodal_content(chat_template, image_token=image_token)
    needs_generation = _GENERATION_RE.search(chat_template) is None

    if not needs_content_renderer and not needs_generation:
        return ChatTemplateInjectionResult(chat_template=chat_template)

    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        message_name = match.group("message")
        content = f"render_tinyllava_content({message_name}.content)"
        if not needs_generation:
            return "{{- " + content + " -}}"

        return (
            "{%- if "
            + message_name
            + ".role == 'assistant' -%}"
            "{% generation %}"
            "{{- "
            + content
            + " -}}"
            "{% endgeneration %}"
            "{%- else -%}"
            "{{- "
            + content
            + " -}}"
            "{%- endif -%}"
        )

    patched = _CONTENT_EXPR_RE.sub(replace, chat_template)
    warnings: list[str] = []

    patched, print_replacements = _inject_into_content_print_blocks(
        patched,
        add_generation=needs_generation,
        add_content_renderer=needs_content_renderer,
    )
    replacements += print_replacements

    if needs_content_renderer:
        patched, set_replacements = _inject_content_variable_assignments(patched)
        replacements += set_replacements

    if needs_generation:
        patched, variable_replacements = _inject_generation_around_content_variables(patched)
        replacements += variable_replacements

    if replacements == 0:
        if needs_generation and not needs_content_renderer:
            patched, replacements = _inject_generation_around_content_loop(patched)

        if replacements == 0 and needs_content_renderer:
            warnings.append(
                "Could not find a plain `message.content` expression to add multimodal rendering."
            )
        if replacements == 0 and needs_generation:
            warnings.append(
                "Could not find a plain `message.content` expression to add generation blocks."
            )

        if replacements > 0:
            if needs_content_renderer and "render_tinyllava_content" not in chat_template:
                patched = _content_renderer_macro(
                    image_token=image_token,
                    image_placement=image_placement,
                    special_content_handlers=special_content_handlers,
                ) + "\n" + patched

            return ChatTemplateInjectionResult(
                chat_template=patched,
                added_content_renderer=needs_content_renderer,
                added_generation_block=needs_generation,
                warnings=tuple(warnings),
            )

        return ChatTemplateInjectionResult(
            chat_template=chat_template,
            added_content_renderer=False,
            added_generation_block=False,
            warnings=tuple(warnings),
        )

    if "render_tinyllava_content" not in chat_template:
        patched = _content_renderer_macro(
            image_token=image_token,
            image_placement=image_placement,
            special_content_handlers=special_content_handlers,
        ) + "\n" + patched

    return ChatTemplateInjectionResult(
        chat_template=patched,
        added_content_renderer=needs_content_renderer,
        added_generation_block=needs_generation,
        warnings=tuple(warnings),
    )


def _inject_into_content_print_blocks(
    chat_template: str,
    add_generation: bool,
    add_content_renderer: bool,
) -> tuple[str, int]:
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        expr = match.group("expr")
        content_match = _MESSAGE_CONTENT_REF_RE.search(expr)
        if content_match is None:
            return match.group(0)
        if "render_tinyllava_content" in expr:
            return match.group(0)

        replacements += 1
        message_name = content_match.group("message")
        content_ref = content_match.group(0)
        rendered_content = (
            f"render_tinyllava_content({content_ref})"
            if add_content_renderer
            else content_ref
        )
        raw_suffix = expr[content_match.end() :].strip()
        if raw_suffix.startswith("|"):
            rendered_content += raw_suffix
            raw_suffix = ""
        content_print_block = "{{ " + rendered_content + " }}"

        # Split the original print expression into three parts:
        #
        #   {{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' }}
        #      ^ prefix                                 ^ content       ^ suffix
        #
        # We must keep role/header text outside `{% generation %}` while keeping
        # the assistant reply terminator inside it.  For Qwen-style templates
        # this means:
        #   - do not supervise `<|im_start|>assistant\n`
        #   - do supervise `assistant text + <|im_end|>\n`
        #
        # Filters immediately attached to content stay with content:
        #   {{ ' ' + message["content"]|trim + eos_token }}
        # becomes conceptually:
        #   {{ ' ' }}{% generation %}{{ render(... )|trim + eos_token }}{% endgeneration %}
        #
        # This also avoids a previous bug where splitting before `|trim`
        # produced invalid Jinja like `{{ |trim + eos_token }}`.
        prefix_expr = _strip_concat_prefix(expr[: content_match.start()])
        suffix_expr = _strip_concat_suffix(raw_suffix)
        prefix_print_block = _print_expr(prefix_expr)
        suffix_print_block = _print_expr(suffix_expr)

        if add_generation:
            content_print_block = (
                "{%- if "
                + message_name
                + ".role == 'assistant' -%}"
                "{% generation %}"
                + content_print_block
                + suffix_print_block
                + "{% endgeneration %}"
                "{%- else -%}"
                + content_print_block
                + suffix_print_block
                + "{%- endif %}"
            )
            suffix_print_block = ""

        return prefix_print_block + content_print_block + suffix_print_block

    return _PRINT_BLOCK_RE.sub(replace, chat_template), replacements


def _print_expr(expr: str) -> str:
    if not expr:
        return ""
    return "{{ " + expr + " }}"


def _strip_concat_prefix(expr: str) -> str:
    expr = expr.strip()
    if expr.endswith("+"):
        expr = expr[:-1].strip()
    return expr


def _strip_concat_suffix(expr: str) -> str:
    expr = expr.strip()
    if expr.startswith("+"):
        expr = expr[1:].strip()
    return expr


def _inject_content_variable_assignments(chat_template: str) -> tuple[str, int]:
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        message_name = match.group("message")
        content_ref = message_name + match.group("content_ref")
        return (
            match.group("set")
            + f"render_tinyllava_content({content_ref})"
            + match.group("end")
        )

    return _SET_CONTENT_FROM_MESSAGE_RE.sub(replace, chat_template), replacements


def _inject_generation_around_content_variables(chat_template: str) -> tuple[str, int]:
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        expr = match.group("expr")
        if "render_tinyllava_content" in expr:
            return match.group(0)
        if _MESSAGE_CONTENT_REF_RE.search(expr) is not None:
            return match.group(0)
        if _CONTENT_VARIABLE_RE.search(expr) is None:
            return match.group(0)

        replacements += 1
        print_block = "{{ " + expr + " }}"
        return (
            "{%- if message.role == 'assistant' -%}"
            "{% generation %}"
            + print_block
            + "{% endgeneration %}"
            "{%- else -%}"
            + print_block
            + "{%- endif %}"
        )

    return _PRINT_BLOCK_RE.sub(replace, chat_template), replacements


def _inject_generation_around_content_loop(chat_template: str) -> tuple[str, int]:
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        message_name = match.group("message")
        loop = match.group("loop")
        return (
            "{%- if "
            + message_name
            + ".role == 'assistant' -%}"
            "{% generation %}"
            + loop
            + "{% endgeneration %}"
            "{%- else -%}"
            + loop
            + "{%- endif -%}"
        )

    return _CONTENT_LOOP_RE.sub(replace, chat_template), replacements


def _handles_multimodal_content(chat_template: str, *, image_token: str) -> bool:
    if image_token in chat_template or image_token.rstrip("\n") in chat_template:
        return True

    return bool(
        re.search(r"content\s*(?:\.|\[\s*['\"])\s*type", chat_template)
        and re.search(r"['\"]image['\"]", chat_template)
    )


def _content_renderer_macro(
    image_token: str,
    image_placement: ImagePlacement,
    special_content_handlers: tuple[str, ...],
) -> str:
    if image_placement == "images_first":
        body = _images_first_content_lines(
            image_token=image_token,
            special_content_handlers=special_content_handlers,
        )
    else:
        body = _interleaved_content_lines(
            image_token=image_token,
            special_content_handlers=special_content_handlers,
        )

    lines = ["{%- macro render_tinyllava_content(content) -%}", *body, "{%- endmacro %}"]
    return "\n".join(lines)


def _images_first_content_lines(
    *,
    image_token: str,
    special_content_handlers: tuple[str, ...],
) -> list[str]:
    return [
        "    {%- if content is string -%}",
        "        {{- content -}}",
        "    {%- else -%}",
        "        {#- Render all images first, matching the default LLaVA chat template. -#}",
        "        {%- for item in content | selectattr('type', 'equalto', 'image') -%}",
        f"            {{{{- {image_token!r} -}}}}",
        *(_special_content_lines(special_content_handlers, indent="            ")),
        "        {%- endfor -%}",
        "        {#- Render all text next. -#}",
        "        {%- for item in content | selectattr('type', 'equalto', 'text') -%}",
        "            {{- item['text'] + ' ' -}}",
        "        {%- endfor -%}",
        "    {%- endif -%}",
    ]


def _interleaved_content_lines(
    image_token: str,
    special_content_handlers: tuple[str, ...],
) -> list[str]:
    return [
        "    {%- if content is string -%}",
        "        {{- content -}}",
        "    {%- else -%}",
        "        {%- for item in content -%}",
        "            {%- if item['type'] == 'image' -%}",
        f"                {{{{- {image_token!r} -}}}}",
        *(_special_content_lines(special_content_handlers, indent="                ")),
        "            {%- elif item['type'] == 'text' -%}",
        "                {{- item['text'] + ' ' -}}",
        "            {%- endif -%}",
        "        {%- endfor -%}",
        "    {%- endif -%}",
    ]


def _special_content_lines(
    special_content_handlers: tuple[str, ...],
    indent: str,
) -> list[str]:
    lines: list[str] = []
    for handler in special_content_handlers:
        if handler == "gemma3_pan_and_scan":
            lines.extend(
                [
                    indent + "{%- for _ in range(item.get('num_crops', 0)) -%}",
                    indent + "    {{- '<image>\\n' -}}",
                    indent + "{%- endfor -%}",
                ]
            )
        else:
            raise ValueError(f"Unknown special content handler: {handler}.")
    return lines


def _validate_image_placement(image_placement: str) -> None:
    if image_placement not in {"images_first", "interleaved"}:
        raise ValueError(
            "`image_placement` must be either 'images_first' or 'interleaved', "
            f"got {image_placement!r}."
        )


__all__ = [
    "ChatTemplateInjectionResult",
    "inject_tinyllava_anchors",
]
