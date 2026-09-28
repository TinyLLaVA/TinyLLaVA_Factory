# OpenELM

Adapted from Apple's OpenELM implementation previously stored in
`tinyllava/model/llm/openelm.py`.

Source model: https://huggingface.co/apple/OpenELM-270M-Instruct

Apple Machine Learning Research Model is licensed under the Apple Machine
Learning Research Model License Agreement. See the accompanying LICENSE.

Local modifications: split configuration/model modules; lazy Auto loading;
current HF Cache and GenerationMixin support; position-aware rotary embeddings;
standard tied output embeddings; removal of the unused backbone classifier and
legacy registry. Layer parameter names and the shared-embedding architecture
are preserved.
