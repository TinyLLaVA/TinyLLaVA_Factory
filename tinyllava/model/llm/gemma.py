from transformers import GemmaForCausalLM, AutoTokenizer

from . import register_llm

@register_llm('gemma')
def return_gemmaclass():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    return (GemmaForCausalLM, (AutoTokenizer, tokenizer_and_post_load))

