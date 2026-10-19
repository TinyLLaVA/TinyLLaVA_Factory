# Configuration API

For YAML composition, see [Configuration](../guides/configuration.md).
For fields grouped by purpose, see the [parameter reference](../guides/parameters.md).

::: tinyllava.hparams.arguments
    options:
      members: [ModelArguments, DataArguments, TuningArguments, TrainingArgBundle, EvalArguments]

::: tinyllava.hparams.hf
    options:
      members: [TrainingArguments]

::: tinyllava.hparams.training
    options:
      members: [build_training_arguments, prepare_training_config, validate_training_config]

::: tinyllava.utils.config
    options:
      members: [compose_config, load_config, merge_mappings, resolve_interpolations]
