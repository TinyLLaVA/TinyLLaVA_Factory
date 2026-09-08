# Extending TinyLLaVA

Use the active interfaces in the [architecture](../architecture.md). Model
construction, tuning policy, data transformation, and scoring are separate
responsibilities.

| Extension | Entry point |
| --- | --- |
| Language model | Hugging Face text config and `language_model_name_or_path` |
| Vision tower | `tinyllava.model.vision_tower.auto` mappings and image processor registration |
| Connector | `tinyllava.model.connector.auto` config/model mappings |
| Dataset schema | `tinyllava.data.adapters` |
| Chat format | `model.chat_template_path` and Jinja generation blocks |
| Tuning policy | `tinyllava.train.strategy` |
| Evaluation task | `tinyllava.eval.tasks.auto` loader/evaluator mappings |

A connector must implement the current config/model contract and produce
features compatible with the language model's hidden size. Register its config
and model in the canonical auto mappings and test save/load round trips.

A custom chat template must render image placeholders consistently and mark
assistant output with `{% generation %}` blocks. Validate rendered prompts and
assistant masks before training to check which tokens contribute to the loss.

For new datasets, adapt the source schema to normalized messages and image
payloads. Keep dataset-specific path handling in the adapter.

Add API directives to the relevant [reference page](../reference/data.md) when
introducing public interfaces.
