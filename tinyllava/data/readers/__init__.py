"""Physical readers for local training data sources."""

from .json_array import is_json_array, iter_json_array, read_first_json_array_item


__all__ = ["is_json_array", "iter_json_array", "read_first_json_array_item"]
