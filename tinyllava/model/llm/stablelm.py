from transformers import StableLmForCausalLM, AutoTokenizer

from . import register_llm

@register_llm('stablelm')
def return_phiclass():
    def tokenizer_and_post_load(tokenizer):
        return tokenizer
    return (StableLmForCausalLM, (AutoTokenizer, tokenizer_and_post_load))
