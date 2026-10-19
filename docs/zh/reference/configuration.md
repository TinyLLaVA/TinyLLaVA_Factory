# 配置 API

YAML 组合与覆写见[配置方式](../guides/configuration.md)，
字段用途见[参数速查](../guides/parameters.md)。

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
