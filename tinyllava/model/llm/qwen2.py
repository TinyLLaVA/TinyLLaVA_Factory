from transformers import Qwen2ForCausalLM, AutoTokenizer

from . import register_llm

@register_llm('qwen2')
def return_qwen2class():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.unk_token = tokenizer.pad_token
#        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    return Qwen2ForCausalLM, (AutoTokenizer, tokenizer_and_post_load)
