import logging
import os

import torch
from peft.tuners.lora import LoraLayer
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)


def lora_kbit_setting(model, training_args):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    
        
        
def maybe_zero_3(param, ignore_status=False, name=None):
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_state_maybe_zero_3(named_params, keys_to_match=[''], require_grad_only=True):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model, skip_keywords=['connector', 'vision_tower']):
    cls = torch.nn.Linear
    lora_module_names = set()
    skip_keywords = skip_keywords
    for name, module in model.named_modules():
        if any(skip_keyword in name for skip_keyword in skip_keywords) or 'lm_head' in name or 'output_layer' in name or 'head' in name:
            continue
        if isinstance(module, cls):
            names = name.split('.')
            #lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(name)
    # if 'lm_head' in lora_module_names:
    #    lora_module_names.remove('lm_head')
    return list(lora_module_names)
