"""Regression checks for bilingual API presentation."""

from pathlib import Path
import tomllib
import unittest

from check_build import PageParser, reference_style_errors


class ReferenceTests(unittest.TestCase):
    def test_bilingual_handler_options_match(self):
        root = Path(__file__).resolve().parents[1]
        handlers = [
            tomllib.loads((root / name).read_text())["project"]["plugins"]["mkdocstrings"]["handlers"]["python"]
            for name in ("zensical.toml", "zensical.zh.toml")
        ]
        self.assertEqual(handlers[0], handlers[1])
        options = handlers[0]["options"]
        for name in ("parameter_headings", "show_symbol_type_heading", "show_symbol_type_toc"):
            self.assertTrue(options[name])
        self.assertEqual(options["docstring_section_style"], "list")

    def test_missing_api_style_is_reported(self):
        errors = reference_style_errors(PageParser())
        self.assertEqual(len(errors), 7)

    def test_stylesheet_links_are_collected(self):
        parser = PageParser()
        parser.feed('<link rel="stylesheet" href="theme.css"><a href="other.css">Link</a>')
        self.assertEqual(parser.stylesheets, {"theme.css"})

    def test_rendered_api_style_is_detected(self):
        parser = PageParser()
        parser.feed('''
            <h2 class="doc-symbol-heading doc-symbol-class"></h2>
            <a class="doc-symbol-toc doc-symbol-attribute"></a>
            <h3 class="doc-symbol-parameter"
                id="tinyllava.eval.generation.make_user_message(text)"></h3>
            <h3 id="tinyllava.eval.tasks.loader_base.GenerationExample.question_id"></h3>
        ''')
        self.assertEqual(reference_style_errors(parser), [])


if __name__ == "__main__":
    unittest.main()
