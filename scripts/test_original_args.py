from dataclasses import dataclass, field, fields
from packaging import version
import pathlib
import draccus

import tokenizers
import transformers


from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments) 
    return model_args

def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args

def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args

def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args


@dataclass
class AllArguments(TrainingArguments):
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)


def train():
    
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()

    # this is just so we have a consistent output with test_new_args.py
    # for straight comparison
    all_arguments = AllArguments(model=model_arguments, data=data_arguments, output_dir=training_arguments.output_dir)
    for field in fields(TrainingArguments):
        setattr(all_arguments, field.name, getattr(training_arguments, field.name))

    draccus.dump(all_arguments, open("config_original.yaml", "w"))
    
    # all_arguments = draccus.parse(config_class=AllArguments)
    # model_arguments, data_arguments, training_arguments = (all_arguments.model, 
    #                                                        all_arguments.data, 
    #                                                        all_arguments.training)
    
    

if __name__ == "__main__":
    train()
