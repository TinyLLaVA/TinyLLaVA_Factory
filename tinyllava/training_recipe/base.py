import os
import torch

from ..utils import *
from ..model import *

class BaseTrainingRecipe:

    def __init__(self, training_arguments):
        self.training_arguments = training_arguments

    
    def __call__(self, model):
        model = self.training_model_converse(model)
        model = self.tune_type_setting(model)
        model.config.tune_type_connector = self.training_arguments.tune_type_connector
        model.config.tune_type_vision_tower = self.training_arguments.tune_type_vision_tower
        model.config.tune_type_llm = self.training_arguments.tune_type_llm
        model.config.tune_vision_tower_from_layer = self.training_arguments.tune_vision_tower_from_layer
        return model
    
 
    def add_args(self, model_args):
        llm_dtype = (torch.float16 if self.training_arguments.fp16 else (torch.bfloat16 if self.training_arguments.bf16 else torch.float32))
        model_args['llm'].update(dict(torch_dtype=llm_dtype))
        if self.training_arguments.pretrained_model_path is not None:
            model_args['llm'].update(dict(pretrained_llm_path=os.path.join(self.training_arguments.pretrained_model_path, 'language_model')))
            model_args['vision_tower'].update(dict(pretrained_vision_tower_path=os.path.join(self.training_arguments.pretrained_model_path, 'vision_tower')))
            model_args['connector'].update(dict(pretrained_connector_path=os.path.join(self.training_arguments.pretrained_model_path, 'connector')))
        return model_args
            
    def tune_type_setting(self, model):
        model = self._llm_tune_type_setting(model)
        model = self._vision_tower_tune_type_setting(model)
        model = self._connector_tune_type_setting(model)
        return model    
        
        
        
    def _llm_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_llm.lower()
        assert tune_type in ('frozen', 'full', 'lora', 'qlora'), f'tune_type {tune_type} not supported in this training recipe!'
        if tune_type == 'full':
            model.language_model.requires_grad_(True)
        elif tune_type == 'frozen':
            model.language_model.requires_grad_(False)
        self.support_gradient_checkpoint(model.language_model, self.training_arguments.gradient_checkpointing)
        return model
        
    def _vision_tower_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_vision_tower.lower()
        assert tune_type in ('frozen', 'full', 'partially-tune', 'lora', 'qlora'), f'tune_type {tune_type} not supported in this training recipe!'
        if tune_type == 'full':
            model.vision_tower.requires_grad_(True)
        elif tune_type == 'frozen':
            model.vision_tower.requires_grad_(False)         
        elif tune_type == 'partially-tune':
            #--------------------------------------------
            #--------------------------------------------
            #TODO gradient checkpointing related???
            #--------------------------------------------
            #--------------------------------------------
            from_layer = self.training_arguments.tune_vision_tower_from_layer
            if from_layer > -1:
                log(f'Tune the vision tower from layer {from_layer}!')
                for n, p in model.vision_tower.named_parameters():
                    if 'vision_model.encoder.layers.' in n: #TODO not sure if other visual encoders contain 'vision_model.encoder.layers.'
                        layer_id = int(n.split('vision_model.encoder.layers.')[-1].split('.')[0])
                        if layer_id >= from_layer:
                            p.requires_grad = True
                        else:
                            p.requires_grad = False
                    else:
                        p.requires_grad = False
        #self.support_gradient_checkpoint(model.vision_tower._vision_tower, self.training_arguments.gradient_checkpointing)
        return model
        
    def _connector_tune_type_setting(self, model):
        tune_type = self.training_arguments.tune_type_connector.lower()
        assert tune_type in ('frozen', 'full', 'lora', 'qlora'), f'tune_type {tune_type} not supported in this training recipe!'   
        if tune_type == 'full':
            for p in model.connector.parameters():
                p.requires_grad = True
        elif tune_type == 'frozen':
            for p in model.connector.parameters():
                p.requires_grad = False
        return model
    
    
        
    def training_model_converse(self, model):
        return model
        
    
    def save(self, model, trainer):
        model.config.use_cache = True
        #save tokenizer       
        model.tokenizer.save_pretrained(self.training_arguments.output_dir)
        #save entire model config
        model.config.save_pretrained(self.training_arguments.output_dir, from_pt=True)
        #save trainer
        trainer.save_state()

        if 'finetune' in self.training_arguments.output_dir and self.training_arguments.pretrained_model_path is not None: # for finetune stage
            if trainer.deepspeed:
                torch.cuda.synchronize()
            trainer.save_model(self.training_arguments.output_dir)
            return
        
        #the followings are for pretrain stage
        #save language model
        language_model_state_dict = get_state_maybe_zero_3(model.language_model.named_parameters(), [''], False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            language_model_output_dir = os.path.join(self.training_arguments.output_dir, 'language_model')
            os.makedirs(language_model_output_dir, exist_ok=True)
            language_model_output_path = os.path.join(self.training_arguments.output_dir, 'language_model/pytorch_model.bin')
            torch.save(language_model_state_dict, language_model_output_path)
            model.config.text_config.save_pretrained(language_model_output_dir, from_pt=True)
        #save vision tower
        vision_tower_state_dict = get_state_maybe_zero_3(model.vision_tower._vision_tower.named_parameters(), [''], False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision_tower_output_dir = os.path.join(self.training_arguments.output_dir, 'vision_tower')
            os.makedirs(vision_tower_output_dir, exist_ok=True)
            vision_tower_output_path = os.path.join(self.training_arguments.output_dir, 'vision_tower/pytorch_model.bin')
            torch.save(vision_tower_state_dict, vision_tower_output_path)
            if isinstance(model.vision_tower._vision_tower, PreTrainedModel):
                model.vision_tower._vision_tower.config.save_pretrained(vision_tower_output_dir, from_pt=True)
        #save connector
        connector_state_dict = get_state_maybe_zero_3(model.connector.named_parameters(), [''], False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            connector_output_dir = os.path.join(self.training_arguments.output_dir, 'connector')
            os.makedirs(connector_output_dir, exist_ok=True)
            connector_output_path = os.path.join(self.training_arguments.output_dir, 'connector/pytorch_model.bin')
            torch.save(connector_state_dict, connector_output_path)
    

    def load(self, model, model_args={}):
        if not ('lora' in self.training_arguments.pretrained_model_path and os.path.exists(os.path.join(self.training_arguments.pretrained_model_path, 'adapter_config.json'))): # loading model for non-lora/non-qlora pretraining
            model.load_llm(**model_args['llm'])
            model.load_vision_tower(**model_args['vision_tower'])
            model.load_connector(**model_args['connector'])
        else:
            model.language_model = model.language_model.from_pretrained(model_args['llm']['model_name_or_path'],attn_implementation='flash_attention_2',torch_dtype=model_args['llm']['torch_dtype'])
            model.load_vision_tower(**model_args['vision_tower'])
            model.load_connector(**model_args['connector'])
            model.to(model_args['llm']['torch_dtype'])
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, self.training_arguments.pretrained_model_path)
            print('Merging LoRA weights...')
            # model = model.merge_and_unload()
            # print('Model is loaded...')

        return model
        
    
    def support_gradient_checkpoint(self, model, gradient_checkpointing=False):
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        if gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        
       
