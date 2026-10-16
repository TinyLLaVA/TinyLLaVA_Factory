# Migrating to v2: purpose, design, and community roadmap

TinyLLaVA Factory is preparing a v2 foundation for **observable, reproducible
small multimodal experiments in university laboratories**. The intended workflow
connects a sample's data source, processed inputs, model behavior, evaluation
result, and controlled interventions. Training and component experiments remain
part of that workflow.

This page explains the transition before integration into `main`. It separates
implemented interfaces from the proposed research tooling, describes migration
costs, and sets out the evidence needed to assess the changes. The ecosystem
references were reviewed in September 2026.

## Continuing the original purpose

The original [TinyLLaVA Factory paper](https://arxiv.org/abs/2405.11788) made
modularity, accessibility, and reproducibility central goals. Its decomposition
into a vision encoder, connector, and language model remains useful for studying
small VLMs. v2 retains that structure and updates the surrounding software
contracts so experiments can be inspected and maintained as the ecosystem grows.

Our direction is a workbench for controlled component experiments,
sample-level observation, and interpretation. For a laboratory with limited
compute, useful questions include:

- How do combinations of vision encoders, connectors, and language models
  compare on task metrics under a controlled training and evaluation protocol?
- How do preprocessing choices such as resolution, cropping, and feature
  selection affect those results, with input alignment, truncation, and
  supervision masks checked for each configuration?
- Which examples improve or regress after a change?
- Does a proposed explanation survive an intervention on the relevant evidence?
- What accuracy improvement justifies extra visual tokens, memory, or latency?

The immediate migration adopts the Hugging Face ecosystem's advanced engineering
designs: configuration-driven Auto dispatch, lazy imports, composable model
configuration, and standard model/processor loading and serialization. This
provides a foundation highly compatible with Transformers and Hub assets for
these experiments, while retaining project-specific component extension points.
The integrated inspection and intervention tools described below are the next
development stage.

## What changed in the wider ecosystem

Recent work expands the questions that a small-model experiment framework must
help investigate. The following examples motivate our priorities; they are not
models already supported by this repository.

| Development | Evidence | Implication for TinyLLaVA |
| --- | --- | --- |
| Visual encoding cost matters alongside language-model size | [FastVLM, CVPR 2025](https://github.com/apple-aiml-research/ml-fastvlm) studies efficient high-resolution encoding; [SigLIP 2, 2025](https://arxiv.org/abs/2502.14786) advances localization and dense features | Measure encoder cost, visual-token count, and answer quality together; expose feature and connector choices |
| Small VLMs extend to video | [SmolVLM2, February 2025](https://huggingface.co/blog/smolvlm2) includes 256M, 500M, and 2.2B image/video models | Plan for frame selection, timestamps, and token budgets in processor and diagnostic interfaces |
| Compact models target fine visual detail | [North Micro Vision, August 2026](https://huggingface.co/blog/CohereLabs/meet-north-micro-vision-instruct) presents a 2.4B model with native-resolution inputs for documents and other visual tasks | Inspect image geometry and evidence loss rather than treating every image as a fixed token grid |
| Evaluation and intervention tools have become reusable building blocks | [LMMS-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [NNsight](https://nnsight.net/) provide multimodal evaluation and model tracing/intervention tools | Connect component experiments with sample-level evaluation and interventions |

[Transformers multimodal chat templates](https://huggingface.co/docs/transformers/en/chat_templating_multimodal)
also place conversation formatting and multimodal preprocessing behind processor
interfaces. Aligning with these contracts makes saved input behavior easier to
reuse across training, evaluation, and interactive generation.

Our contribution should be the connection between these activities: composing
model and preprocessing configurations, checking their input contracts, comparing
metrics and resource costs, and inspecting the samples and representations that
explain the differences. Evaluation backends, profilers, and intervention
libraries are candidates for integration. Such integrations still require
explicit adapters and validation.

### A concrete component example: MoF

Interleaved MoF in [*Eyes Wide Shut?*](https://arxiv.org/html/2401.06209)
projects CLIP and DINOv2 features separately and interleaves them in spatial
order. The project encapsulates dual-branch encoding in the vision tower and
projection/interleaving in the connector. The connector declares its output
length so the shared processor can align image placeholders. Common contracts
connect model assembly and input processing, providing a basis for comparing
component combinations.

The current implementation shares image inputs and requires compatible feature
grids and other branch properties. New preprocessing or fusion designs still
need contract validation; the modular interfaces centralize that adaptation.

## Why the foundation needs to change

### Discovery by importing every implementation

The early component factories scanned their package directories and imported
Python modules so decorators could populate local registries. Language and vision
selection then searched registered names using substrings of a model path.

That design had concrete costs:

- Package discovery executed imports for implementations an experiment might
  never use. Their transitive dependencies increased the startup dependency
  surface and could make unrelated imports fail.
- Adding implementations expanded import work. Fresh training or evaluation
  processes repeated it, even when they selected the same small subset.
- Substring selection scanned the registry and could produce ambiguous,
  iteration-order-dependent matches. Renaming a directory could affect dispatch.

The dictionary itself was inexpensive. The relevant performance concern was
**eager discovery and its transitive imports**, with a separate correctness
problem in name matching. This was startup work, not a per-token registry lookup
inside generation. We have no controlled before/after latency or memory study
from which to claim a numerical speedup or improved training throughput.

v2 uses config/model mappings and lazy implementation imports. Small registration
and configuration modules may still load eagerly; selecting a component imports
its implementation when needed. Regression tests check dispatch and deferred
imports. Performance measurements should separately record cold package import,
configuration construction, and weight loading, along with dependency versions,
resident memory, and process count. For example, Python's `-X importtime` can
identify import costs; it does not measure GPU inference speed.

### Ownership of experiment behavior

In v1, recipes and command-specific code also carried loading, tuning, saving,
and input-format decisions. Extending an experiment could require coordinated
changes in several entry points. Model weights alone did not identify all the
input and execution choices needed to reproduce an answer.

v2 assigns those decisions to narrower owners:

| Responsibility | v2 owner | Why it matters |
| --- | --- | --- |
| Architecture and component selection | Composite `TinyLlavaConfig` and component Auto mappings | A saved configuration describes the assembled model |
| Tokenization, image preprocessing, templates, and image placeholders | Processor and image processor | Training and inference can use the same input definition |
| Connector output length | Connector configuration, consumed by the processor | Image placeholders stay aligned with projected features |
| Freezing and parameter-efficient tuning | Tuning strategies | Training policy can change without rewriting model loading |
| Model, data, recipes, stages, and launch settings | Composable YAML presets and experiment runner | Experiment choices are reviewable and reusable |
| Dataset source and sample conversion | Hugging Face Datasets loading and data adapters | Storage format is separated from conversation encoding |
| Checkpoint generation and task scoring | Shared generation utilities and evaluation adapters | Input behavior is shared while task-specific scoring remains explicit |

See [Architecture](architecture.md) for module boundaries and
[Extending](guides/extending.md) for the actual registration interfaces.

## Technical decisions and their tradeoffs

### Retain HF-style Auto dispatch and namespace isolation

Native Transformers components use upstream implementations where compatible.
Project components retain local mappings and lazy packages. This preserves
extension points without carrying copies of every supported backbone.

Project-specific model types use namespaces such as `mlp__tlf_connector` and
`mof__tlf_vision_tower`. The full model type remains the module-cache key; the
suffix is removed only when resolving the local Python import path. This keeps
a project type distinct from an upstream type with the same base name.

The local mapping classes specialize module resolution and inherit the HF
mapping protocol. They depend on some private Transformers Auto internals, so
Transformers upgrades require dispatch, serialization, and lazy-import regression
checks. Project and upstream runtime registration are distinct entry points;
registering a class in one does not automatically synchronize the other.

Auto dispatch establishes how a class is selected. A new backbone must still
satisfy the feature, hidden-size, processor, and generation contracts of the
composite model. An upstream Auto entry alone does not establish compatibility
with every multimodal architecture.

### Make the processor extensible and the length contract explicit

`ComponentProcessorMixin` persists vision/connector configuration and exposes
opt-in length helpers without inheriting a particular HF processor.
`TinyLlavaProcessor(ComponentProcessorMixin, LlavaProcessor)` owns construction
from a model, LLaVA image-token expansion, and multimodal token estimates.
Other processors can reuse the mixin while retaining their native placeholder,
video, and metadata handling. Each concrete processor decides where to apply
component rules; a native merger must not be counted twice. Custom processor
classes are selected through the project Auto interface. The separate image
processor extension point is retained.

A connector owns its output-length rule. MLP and identity connectors preserve
sequence length; query-based connectors can produce a fixed query count; the
current MoF connector interleaves its two projected feature streams. The shared
processor consumes that rule instead of assuming every connector implements
`N → N` or branching on connector names. This contract is also a prerequisite
for future visual-token inspection.

These are extensibility contracts. Video decoding, frame/time alignment, arbitrary
multi-image layouts, and native-resolution architectures need additional
implementations and tests before they become supported workflows.

### Preserve input behavior with the checkpoint

An explicit `model.chat_template_path` takes precedence over the tokenizer's
template. There is no universal silent fallback when both are missing.
The Phi-2 and OpenELM presets therefore explicitly select the bundled HF LLaVA
template, sourced from
[`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/main/chat_template.jinja).
It defines a new experiment's prompt format; it does not establish the historical
format of every checkpoint using those language models.

Legacy templates remain explicit recipe choices. In particular,
`pretrain_legacy` is an image-to-answer pretraining format that omits the user
question, so it is unsuitable as a general instruction-template fallback.
Before training, inspect rendered questions, image placeholders, assistant loss
masks, and termination behavior. Save and reload the processor with the model so
evaluation uses the intended template.

### Compose experiments and chain complete checkpoints

Presets separate model, dataset, and training policy. An experiment specifies
ordered stages, and subsequent stages load the preceding composite checkpoint.
Stage names are labels; their recipes determine training behavior. Selecting a
later stage does not train missing prerequisites automatically.

Dataset sources can now use Hub datasets and local files through Hugging Face
Datasets. A source loader determines how records are obtained; an adapter
converts records into the expected samples. Supporting a file format does not
mean every dataset schema or modality already has an adapter.

## Roadmap: from interfaces to an experiment workbench

The following stages express priorities and acceptance criteria, not a release
calendar or a list of completed features.

| Stage | Planned capability | Evidence of completion |
| --- | --- | --- |
| Foundation — this migration | Shared model/processor contracts, composable experiments, dataset loading, generation and evaluation interfaces | Interface regression checks plus representative checkpoint and experiment validation |
| Sample inspection and replay | Stable sample/run identifiers; data provenance; template, mask, token and image-geometry inspection; minimal trace records | A failed prediction can be replayed with the same assets and processing choices |
| Diagnosis and comparison | Per-component timing/memory, sample-level result differences, capability slices, data-quality reports | An experiment report links regressions and resource costs to inspectable samples |
| Explanations and interventions | Semantic access to vision, connector and language representations; occlusion, token ablation, activation intervention and controls | Visualizations are paired with intervention outcomes and clearly scoped interpretations |
| Broader modalities and recipes | Validated multi-image/video processing and targeted research recipes | Modality-specific alignment, batching, replay and evaluation checks pass |

A proposed trace record should link sample identity, model and data versions,
resolved configuration, media identity, token spans, generation settings,
predictions, metrics, and capture/intervention settings. Lightweight statistics
should be the default. Full attention matrices and activations require explicit
capture settings and measured overhead.

Attention views can help generate hypotheses. A causal explanation needs
interventions and controls: for example, compare evidence occlusion with matched
irrelevant-region occlusion, or compare an activation intervention with a control
at the same layer. The planned tools should report sensitivity and failure cases
rather than treating an attractive heatmap as validation.

### Refresh data and evaluation through traceable experiments

The existing LLaVA-style datasets and benchmark recipes remain useful historical
baselines. New experiments need coverage of documents, fine visual evidence,
grounding, and eventually temporal reasoning. Dataset age alone is insufficient
to decide whether to retain or replace it.

Candidate sources include task-specific subsets of
[The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) and the
image/video and pointing resources released with
[Molmo 2](https://github.com/allenai/molmo2). These are research inputs to evaluate,
not newly bundled training presets. Selection should record the source and
version, task coverage, available media, split, filtering, and sampling policy;
then check overlap with evaluation data. A larger mixture needs an ablation
against a fixed baseline before becoming a recommended recipe.

Keep historical evaluation for continuity, add targeted failure slices, and
explore adapters to established evaluation toolkits. Report per-task and
per-sample changes with the prompt, generation budget, scoring procedure, and
resource cost. The current built-in task set is documented in
[Evaluation](guides/evaluation.md); broader benchmark integration is roadmap work.

## Compatibility and migration

v2.0 is unreleased. Intermediate configuration and internal API changes do not
retain compatibility branches. Final breaking changes are recorded in the
[v2.0 changelog](https://github.com/TinyLLaVA/TinyLLaVA_Factory/blob/develop/CHANGELOG.md); configuration is validated against the current schema.

v2 changes extension and configuration interfaces. Original experiments remain
available through `legacy/v1`; use a separate environment when reproducing them.
The historical `doc` branch archives the v1 Sphinx documentation.

| Existing asset | Required action |
| --- | --- |
| v1 shell command or recipe | Translate model, data, training and launch choices into presets; check the resolved configuration |
| Custom component factory | Register a config/model pair in the relevant Auto interface and satisfy its component contract |
| Custom conversation formatter | Supply an appropriate processor/template and verify rendered tokens and supervision masks |
| Legacy checkpoint | Check architecture/configuration and processor assets; validate known weight-key conversions and a save/load round trip |
| Old optimizer/scheduler/distributed state | Verify separately; model-weight conversion does not establish resume compatibility |
| Published benchmark score | Retain its original experimental context and re-evaluate migrated checkpoints before claiming equivalence |

The loader provides weight-key renaming for known legacy language-model,
vision-tower, and MLP projector layouts. It does not reconstruct missing configs,
tokenizers, image processors, or templates, nor convert arbitrary custom
connector architectures. Current-stack training checkpoints are the intended
resume path. Starting a new stage from model weights and resuming an interrupted
optimizer run are different operations.

### Suggested migration sequence

1. **Preserve the baseline.** Record the original environment, assets, input
   formatting, data split, training settings, generation settings, and a small
   fixed set of outputs. Keep the original checkpoint intact.
2. **Choose the new experiment.** Start from `configs/experiments/`, with presets
   in `configs/models/`, `configs/data/train/`, and `configs/training/`. Review the
   resolved configuration before loading weights:

   ```bash
   python -m tinyllava.cli train --config configs/experiments/phi.yaml --dry-run
   ```

3. **Validate inputs and features.** Inspect samples after adaptation and
   processing. Check the user question, assistant mask, image-token count,
   feature-selection policy, connector length, and truncation.
4. **Validate checkpoint behavior.** Load, save, and reload the model and
   processor. Compare fixed-input results under the same precision and generation
   settings. Investigate missing or unexpected weights before training.
5. **Run a small experiment.** Check trainable parameters, loss, finite gradients,
   effective global batch, and stage output. A dry run validates configuration,
   not model compatibility or numerical behavior.
6. **Evaluate and record differences.** Use matching task definitions and scoring.
   Document intentional changes to templates, preprocessing, masks, data, or
   generation rather than attributing all differences to the architecture.

See [Installation](guides/installation.md), [Training](guides/training.md),
[Data](guides/data.md), and [Inference](guides/inference.md) for executable workflows.
The Qwen2 base legacy experiment retains explicit historical prompt choices.

## Evidence expected before the transition is complete

Interface tests and documentation builds are necessary checks. Representative
training, checkpoint reload, generation, and evaluation runs are also needed to
establish practical readiness. Claims of reproduction require matched data and
experimental settings; claims of speed or memory improvements require measured
baselines. The original paper's scores remain results of its original experiments.

Community reports are most useful when they include a minimal configuration,
component and dataset identifiers, environment versions, a reproducible sample,
and the expected versus observed behavior. Contributions that add a component,
processor, or diagnostic should include a small contract test and explain the
experimental question it enables. This makes each extension part of an
inspectable research workflow.
