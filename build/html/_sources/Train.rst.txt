Train
====================

Here's an example for training a LMM using Phi-2.

• Replace data paths with yours in ``scripts/train/train_phi.sh``

• Replace ``output_dir`` with yours in ``scripts/train/pretrain.sh``

• Replace ``pretrained_model_path`` and ``output_dir`` with yours in ``scripts/train/finetune.sh``

• Adjust your GPU ids (localhost) and ``per_device_train_batch_size`` in ``scripts/train/pretrain.sh`` and ``scripts/train/finetune.sh``

.. code-block:: bash

   bash scripts/train/train_phi.sh

Important hyperparameters used in pretraining and finetuning are provided below.

+----------------+--------------------+-----------------+--------------+
| Training Stage | Global Batch Size  |  Learning rate  | conv_version | 
+================+====================+=================+==============+
|   Pretraining  |        256         |      1e-3       |   pretrain   |
+----------------+--------------------+-----------------+--------------+
|   Finetuning   |        128         |      2e-5       |     phi      |
+----------------+--------------------+-----------------+--------------+

**Tips:**

• Global Batch Size = num of GPUs * ``per_device_train_batch_size`` * ``gradient_accumulation_steps``, we recommand you always keep global batch size and learning rate as above except for lora tuning your model.

• ``conv_version`` is a hyperparameter used for choosing different chat templates for different LLMs. In the pretraining stage, ``conv_version`` is the same for all LLMs, using ``pretrain``. In the finetuning stage, we use

    • ``phi`` for Phi-2, StableLM, Qwen-1.5

    • ``llama`` for TinyLlama, OpenELM

    • ``gemma`` for Gemma