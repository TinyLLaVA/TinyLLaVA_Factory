"""Check navigation pages and rendered API anchors after a Zensical build.

Run from any directory with Python 3.11+; no training dependencies are needed.
"""

from html.parser import HTMLParser
import argparse
from pathlib import Path
import re
import tomllib
from urllib.parse import unquote, urlsplit


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = set()
        self.language = None
        self.classes = set()
        self.stylesheets = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.classes.update(attributes.get("class", "").split())
        if tag == "link" and "stylesheet" in attributes.get("rel", "").split():
            self.stylesheets.add(attributes.get("href", ""))
        if tag == "html":
            self.language = attributes.get("lang")
        if tag in {"a", "link"} and attributes.get("href"):
            self.links.add(attributes["href"])
        if tag in {"img", "script"} and attributes.get("src"):
            self.links.add(attributes["src"])
        for name, value in attrs:
            if name == "id":
                self.ids.add(value)


def nav_paths(items):
    for item in items:
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, list):
                    yield from nav_paths(value)
                else:
                    yield value


def reference_style_errors(parser):
    """Check representative API badges and parameter/attribute anchors."""
    required_classes = {
        "doc-symbol-heading", "doc-symbol-toc", "doc-symbol-class",
        "doc-symbol-attribute", "doc-symbol-parameter",
    }
    required_ids = {
        "tinyllava.eval.generation.make_user_message(text)",
        "tinyllava.eval.tasks.loader_base.GenerationExample.question_id",
    }
    return (
        [f"Missing API style class: {name}" for name in sorted(required_classes - parser.classes)]
        + [f"Missing API detail anchor: {name}" for name in sorted(required_ids - parser.ids)]
    )


def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--config", default="zensical.toml")
    arguments.add_argument("--base-path", default="", help="Published URL prefix, e.g. /repository")
    options = arguments.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / options.config).read_text())["project"]
    docs = root / project["docs_dir"]
    site = root / project["site_dir"]
    combined_site = root / "site"
    failures = []
    pages = list(nav_paths(project["nav"]))
    for name in pages:
        source = docs / name
        if not source.is_file():
            failures.append(f"Missing navigation source: {source}")
            continue
        relative = Path(name).with_suffix(".html")
        if project.get("use_directory_urls", True) and relative.name != "index.html":
            relative = relative.with_suffix("") / "index.html"
        output = site / relative
        if not output.is_file() or output.stat().st_size == 0:
            failures.append(f"Missing or empty page: {output}")
            continue
        parser = PageParser()
        parser.feed(output.read_text(encoding="utf-8"))
        if name == "reference/evaluation.md":
            failures.extend(f"{relative}: {error}" for error in reference_style_errors(parser))
        if parser.language != project["theme"]["language"]:
            failures.append(f"Incorrect HTML language in {relative}: {parser.language}")
        for alternate in project["extra"]["alternate"]:
            if alternate["link"] not in parser.links:
                failures.append(f"Missing language selector in {relative}: {alternate['link']}")
        stylesheet_text = []
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            path = unquote(url.path)
            if path.startswith("/") and options.base_path:
                prefix = options.base_path.rstrip("/") + "/"
                if not path.startswith(prefix):
                    failures.append(f"Link escapes published base path in {relative}: {link}")
                    continue
                path = "/" + path[len(prefix):]
            target = combined_site / path.lstrip("/") if path.startswith("/") else output.parent / path
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                failures.append(f"Missing local link or asset in {relative}: {link}")
            elif name == "reference/evaluation.md" and link in parser.stylesheets:
                stylesheet_text.append(target.read_text(encoding="utf-8"))
        if name == "reference/evaluation.md":
            css = "\n".join(stylesheet_text)
            for kind in ("attribute", "parameter", "class"):
                if f"doc-symbol-{kind}" not in css:
                    failures.append(f"Missing API badge stylesheet in {relative}: {kind}")
        for symbol in re.findall(r"^::: +([\w.]+)", source.read_text(), re.MULTILINE):
            if symbol not in parser.ids:
                failures.append(f"Missing rendered API anchor in {relative}: {symbol}")
    if failures:
        raise SystemExit("Documentation validation failed:\n" + "\n".join(failures))
    print(f"Validated {len(pages)} pages: API anchors and badges, language selectors, local links and assets.")


if __name__ == "__main__":
    main()
