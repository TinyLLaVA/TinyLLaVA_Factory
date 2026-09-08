# 配置 API

训练与评测共享参数定义。训练参数扩展 HF TrainingArguments，以下列出项目定义的字段。

::: tinyllava.utils.config
    options:
      members: [parse_train_config, build_train_arguments, parse_eval_config, build_eval_arguments, load_connector_config]

::: tinyllava.utils.arguments
    options:
      members: [ModelArguments, DataArguments, TrainingArguments, EvalModelArguments, EvalDataArguments, EvalGenerationArguments, EvalRuntimeArguments, EvalOutputArguments, EvalArguments]
