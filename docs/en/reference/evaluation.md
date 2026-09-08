# Evaluation API

Generation helpers return decoded text. Loaders prepare generation examples;
evaluators score or export predictions. See [Evaluation](../guides/evaluation.md).

::: tinyllava.eval.generation
    options:
      members: [make_user_message, prepare_generation_inputs, generate_response, generate_responses]

::: tinyllava.eval.tasks.loader_base.GenerationExample

::: tinyllava.eval.tasks.loader_base.DatasetLoader

::: tinyllava.eval.tasks.evaluation_base.DatasetEvaluation
