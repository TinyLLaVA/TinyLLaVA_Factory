"""Identity connector configuration."""

from ..configuration_base import BaseConnectorConfig


class IdentityConnectorConfig(BaseConnectorConfig):
    model_type = "identity__tlf_connector"


__all__ = ["IdentityConnectorConfig"]
