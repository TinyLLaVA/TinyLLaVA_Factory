# Training API

Strategies apply tuning policies and save the model. The optional sampler
groups samples by modality and length.

::: tinyllava.train.strategy.base.BaseTrainingStrategy

::: tinyllava.train.modality_trainer.TinyLlavaTrainer

::: tinyllava.train.modality_trainer.ModalityLengthGroupedSampler

::: tinyllava.utils.checkpoint
    options:
      members: [find_last_complete_checkpoint, resolve_resume_checkpoint]
