"""Check deployment configuration without a watcher, network, or GPU runtime."""

import argparse
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest
from unittest.mock import patch

import build


class BuildTests(unittest.TestCase):
    def run_build(self, url=None, fail=False):
        root = Path(build.__file__).resolve().parents[1]
        originals = {name: (root / name).read_bytes()
                     for name in ("zensical.toml", "zensical.zh.toml")}
        calls, configs, temporary_paths = [], [], []

        def run(command, **kwargs):
            self.assertTrue(kwargs["check"])
            self.assertEqual(kwargs["cwd"], root)
            calls.append(command)
            if command[1] == "build":
                path = Path(command[-1])
                configs.append(tomllib.loads(path.read_text())["project"])
                if url:
                    temporary_paths.append(path)
                if fail:
                    raise subprocess.CalledProcessError(1, command)

        argv = ["build.py"] + (["--base-url", url] if url else [])
        with patch.object(sys, "argv", argv), patch.object(build.subprocess, "run", side_effect=run):
            if fail:
                with self.assertRaises(subprocess.CalledProcessError):
                    build.main()
            else:
                build.main()
        for path in temporary_paths:
            self.assertFalse(path.exists())
        for name, content in originals.items():
            self.assertEqual((root / name).read_bytes(), content)
        return calls, configs

    def test_local_build_order(self):
        calls, configs = self.run_build()
        self.assertEqual([c["site_dir"] for c in configs], ["site", "site/zh"])
        self.assertEqual(len(calls), 4)
        self.assertTrue(all("check_build.py" in c[1] for c in calls[2:]))

    def test_repository_subpath(self):
        calls, configs = self.run_build("https://example.github.io/project/")
        self.assertEqual([c["site_url"] for c in configs],
                         ["https://example.github.io/project/", "https://example.github.io/project/zh/"])
        for config in configs:
            self.assertEqual([a["link"] for a in config["extra"]["alternate"]], ["/project/", "/project/zh/"])
        self.assertEqual(calls[-1][-2:], ["--base-path", "/project"])

    def test_custom_domain(self):
        _, configs = self.run_build("https://docs.example.org")
        self.assertEqual(configs[1]["site_url"], "https://docs.example.org/zh/")
        self.assertEqual(configs[0]["extra"]["alternate"][0]["link"], "/")

    def test_cleanup_on_build_failure(self):
        calls, _ = self.run_build("https://example.org/project", fail=True)
        self.assertEqual(len(calls), 1)

    def test_reject_invalid_urls(self):
        for value in ("", "relative/path", "file:///tmp/site", "https://user:pass@example.org",
                      "https://example.org/?query=1", "https://example.org/#anchor"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                build.deployment_url(value)


if __name__ == "__main__":
    unittest.main()
