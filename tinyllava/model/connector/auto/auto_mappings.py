from collections import OrderedDict

CONNECTOR_CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("mlp__tlf_connector", "MLPConnectorConfig"),
        ("mof__tlf_connector", "MofConnectorConfig"),
        ("resampler__tlf_connector", "ResamplerConnectorConfig"),
        ("qformer__tlf_connector", "QFormerConnectorConfig"),
        ("identity__tlf_connector", "IdentityConnectorConfig"),
    ]
)

CONNECTOR_MODEL_MAPPING_NAMES = OrderedDict(
    [
        ("mlp__tlf_connector", "MLPConnector"),
        ("mof__tlf_connector", "MofConnector"),
        ("resampler__tlf_connector", "ResamplerConnector"),
        ("qformer__tlf_connector", "QFormerConnector"),
        ("identity__tlf_connector", "IdentityConnector"),
    ]
)
