"""YAML reference composition, resolved worker YAML, and mapping operations."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
_PRESET_DIRS = {
    "model": "models",
    "connector": "connectors",
    "training": "training",
    "data": "data/train",
}


def compose_config(
    path: str | Path,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Compose YAML references while leaving value interpolation unresolved.

    Any mapping can use ``from: path.yaml`` or a list of paths. Sources merge
    left to right, then local fields, then CLI dotlist overrides. Bare filenames
    at preset nodes use the corresponding directory under configs/. A preset name
    is shorthand for its ``from`` mapping; references without a YAML suffix get ``.yaml`` appended.
    Explicit relative paths are anchored to their declaring YAML, or cwd for CLI
    overrides. Lists replace rather than
    concatenate. Value interpolation is deferred until runtime paths are assigned.
    """
    preset_dirs = _PRESET_DIRS
    cache: dict[Path, dict[str, Any]] = {}

    def read(source: Path, location: tuple = ()) -> dict[str, Any]:
        if source not in cache:
            content = source.read_text(encoding="utf-8")
            if not isinstance(
                yaml.compose(content, Loader=yaml.SafeLoader), yaml.MappingNode
            ):
                raise ValueError(f"Config must be a mapping: {source}")
            config = OmegaConf.create(content)
            cache[source] = OmegaConf.to_container(config, resolve=False)
        return _anchor_references(
            cache[source], source.parent, str(source), preset_dirs, location
        )

    def inherit(node: dict, location: tuple, chain: tuple[Path, ...] = ()) -> dict:
        # Expand only this mapping's sources. Nested references must wait until
        # all overlays are merged, so replacing a source cannot leak old fields.
        local = {key: value for key, value in node.items() if key != "from"}
        sources = node.get("from", [])
        sources = [sources] if isinstance(sources, str) else sources
        bases = []
        for reference in sources:
            source = Path(reference)
            if source in chain:
                trace = " -> ".join(map(str, (*chain, source)))
                raise ValueError(f"Circular config reference: {trace}")
            try:
                base = read(source, location)
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"Config reference not found: {source}"
                ) from error
            bases.append(inherit(base, location, (*chain, source)))
        return merge_mappings(*bases, local)

    def expand(
        node: Any, location: tuple = (), ancestors: tuple[dict, ...] = ()
    ) -> Any:
        if isinstance(node, list):
            return [
                expand(value, (*location, index), ancestors)
                for index, value in enumerate(node)
            ]
        if not isinstance(node, dict):
            return node
        if "from" in node:
            # Compare the unexpanded mapping, not just filenames: the same file
            # can occur at different nodes with different, finite local overlays.
            if node in ancestors:
                trace = " -> ".join(str(item["from"]) for item in (*ancestors, node))
                raise ValueError(f"Circular config reference: {trace}")
            ancestors = (*ancestors, node)
            node = inherit(node, location)
        return {
            key: expand(value, (*location, key), ancestors)
            for key, value in node.items()
        }

    path = Path(path).expanduser().resolve()
    config = read(path)
    # Normalize each override before merging: model=phi followed by
    # model.model_max_length=4096 must retain both the reference and the field.
    for override in overrides:
        cli = _anchor_references(
            OmegaConf.to_container(OmegaConf.from_dotlist([override]), resolve=False),
            Path.cwd(),
            "CLI overrides",
            preset_dirs,
        )
        config = merge_mappings(config, cli)
    return expand(config)


def _anchor_references(
    node: Any,
    directory: Path,
    label: str,
    preset_dirs: dict[str, str],
    location: tuple = (),
) -> Any:
    """Expand shorthand and preserve reference origins before mapping overlays."""
    # A stage name is not a preset field, even if it is named "model" or "data".
    field = location[-1] if location else None
    is_stage = len(location) >= 2 and location[-2] == "stages"
    preset = preset_dirs.get(field) if not is_stage else None
    if preset is not None and isinstance(node, str) and "${" not in node:
        node = {"from": node}
    if isinstance(node, list):
        return [
            _anchor_references(value, directory, label, preset_dirs, (*location, index))
            for index, value in enumerate(node)
        ]
    if not isinstance(node, dict):
        return node
    result = {}
    for key, value in node.items():
        if key != "from":
            result[key] = _anchor_references(
                value, directory, f"{label}.{key}", preset_dirs, (*location, key)
            )
            continue
        paths = [value] if isinstance(value, str) else value
        if (
            not isinstance(paths, list)
            or not paths
            or any(
                not isinstance(item, str) or not item.strip() or "${" in item
                for item in paths
            )
        ):
            raise ValueError(
                f"{label}.from must be a literal YAML path or a nonempty list of paths"
            )
        anchored = []
        for item in paths:
            root = (
                CONFIG_ROOT / preset
                if preset and Path(item).name == item
                else directory
            )
            reference = Path(item).expanduser()
            if reference.suffix.lower() not in (".yaml", ".yml"):
                reference = reference.with_name(reference.name + ".yaml")
            anchored.append(str((root / reference).resolve()))
        result[key] = anchored[0] if isinstance(value, str) else anchored
    return result


def resolve_interpolations(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve runtime placeholders after stage paths have been assigned."""
    return OmegaConf.to_container(OmegaConf.create(config), resolve=True)


def merge_mappings(*configs: dict[str, Any] | DictConfig) -> dict[str, Any]:
    """Deep-merge left to right, preserving interpolation for the final pass."""
    return OmegaConf.to_container(OmegaConf.merge(*configs), resolve=False)


def validate_name(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError(f"{label} must be a nonempty name without path separators")
    return value


def validate_fields(
    config: dict[str, Any], sections: Collection[str], metadata: set[str], kind: str
) -> None:
    unknown = set(config) - set(sections) - metadata
    if unknown:
        raise ValueError(f"Unknown {kind} config fields: {sorted(unknown)}")
    for section in sections:
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"{kind}.{section} must be a mapping")


def positive_int(value: Any, name: str) -> int:
    """Validate an integer configuration value, excluding booleans and zero."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Read a mapping, apply field overrides, and resolve interpolation.

    This helper preserves the supplied schema; it does not expand presets or
    supply business defaults.
    """
    config = OmegaConf.load(path)
    if not OmegaConf.is_dict(config):
        raise ValueError(f"Config must be a mapping: {path}")
    if overrides:
        config = merge_mappings(config, OmegaConf.from_dotlist(list(overrides)))
    return resolve_interpolations(config)
