# Training API

Strategies apply tuning policies and save the model. The optional sampler
groups samples by modality and length.

::: tinyllava.train.strategy.base.BaseTrainingStrategy

::: tinyllava.train.algorithms.sft_trainer.TinyLlavaTrainer

::: tinyllava.train.algorithms.sft_trainer.ModalityLengthGroupedSampler

::: tinyllava.utils.checkpoint
    options:
      members: [find_last_complete_checkpoint, resolve_resume_checkpoint]
