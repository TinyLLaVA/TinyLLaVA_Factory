from collections import OrderedDict


TRAINING_STRATEGY_MAPPING_NAMES = OrderedDict(
    [
        ("common", "BaseTrainingStrategy"),
        ("lora", "LoraTrainingStrategy"),
        ("lora_int8", "LoraInt8TrainingStrategy"),
    ]
)
