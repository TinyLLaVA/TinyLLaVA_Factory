"""Build and validate both languages, optionally for a deployed base URL."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
from urllib.parse import urlsplit

import tomli_w


def deployment_url(value):
    url = urlsplit(value)
    if (url.scheme not in {"http", "https"} or not url.netloc
            or url.username or url.password or url.query or url.fragment):
        raise argparse.ArgumentTypeError("Expected an HTTP(S) site URL without credentials, query or fragment")
    return value.rstrip("/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", type=deployment_url)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base_path = urlsplit(args.base_url).path.rstrip("/") if args.base_url else ""
    executable = str(Path(sys.executable).with_name("zensical"))
    configs = []
    with ExitStack() as stack:
        # Keep temporary configs at the repository root so relative source paths
        # retain their meaning. Originals and the local preview URLs stay intact.
        for filename, suffix in (("zensical.toml", ""), ("zensical.zh.toml", "/zh")):
            config = root / filename
            if args.base_url:
                data = tomllib.loads(config.read_text(encoding="utf-8"))
                project = data["project"]
                project["site_url"] = args.base_url + suffix + "/"
                for alternate in project["extra"]["alternate"]:
                    alternate["link"] = base_path + ("/zh/" if alternate["lang"] == "zh" else "/")
                temporary = stack.enter_context(tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=root,
                    prefix=".docs-build-", suffix=".toml",
                ))
                temporary.write(tomli_w.dumps(data))
                temporary.flush()
                config = Path(temporary.name)
            configs.append(config)
            # English clears site/, so building the languages in parallel is unsafe.
            subprocess.run([executable, "build", "--clean", "--strict", "-f", str(config)],
                           cwd=root, check=True)
        # Validate after both builds, including links between language sites.
        for config in configs:
            subprocess.run([sys.executable, str(root / "docs/check_build.py"),
                            "--config", str(config), "--base-path", base_path],
                           cwd=root, check=True)


if __name__ == "__main__":
    main()
