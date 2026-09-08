# Configuration API

Training and evaluation share `tinyllava.utils.arguments`. Custom training
arguments extend HF TrainingArguments. The reference below lists project-defined fields.

::: tinyllava.utils.config
    options:
      members: [parse_train_config, build_train_arguments, parse_eval_config, build_eval_arguments, load_connector_config]

::: tinyllava.utils.arguments
    options:
      members: [ModelArguments, DataArguments, TrainingArguments, EvalModelArguments, EvalDataArguments, EvalGenerationArguments, EvalRuntimeArguments, EvalOutputArguments, EvalArguments]
