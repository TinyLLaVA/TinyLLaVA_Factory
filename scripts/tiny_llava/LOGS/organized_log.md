# V1
# 实验1：tinyllama-standard-data
## 实验时间：2024年2月1日13点37分
## 实验重要参数：
  * LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: No
  * pretrain lr&batch size: 1e-3 256
  * finetune lr&batch size: 2e-5 128
  * data: standard llava-1.5
  * data type: fp16
## 训练策略：
预训练：与LLaVA一致
微调：与LLaVA一致
## 实验结果：
  * GQA: 58.05
  * SQA: 60.24
  * TextVQA: 45.83
  * VQAv2:
  * VizWiz:
## 实验分析：
这个仅是baseline在transformer版本升级后的复现，不应该有什么变化，但是由于tokenizers版本的升级，会出现mismatch，更新train.py的代码后（use_fast=True或使用LLaVA-1.6的补丁）可以兼容升级。
TextVQA的成绩略有下降

# 实验2：stablelm-standard-data
## 实验时间：2024年2月3日22点29分
## 实验重要参数：
  * LLM: stabilityai/stablelm-2-zephyr-1_6b
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: No
  * pretrain lr&batch size: 1e-3 256
  * finetune lr&batch size: 2e-5 128
  * data: standard llava-1.5
  * data type: bf16
## 训练策略：
预训练：与LLaVA-1.5一致
微调：与LLaVA-1.5一致
## 实验结果：
  * GQA: 58.86
  * SQA: 62.82
  * TextVQA: 49.52
  * VQAv2: 74.9
  * VizWiz:
## 实验分析：

# 实验3：tinyllama-standard-data-siglip
## 实验时间：2024年2月9日14点45分
## 实验重要参数：
  * LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  * VT: google/siglip-so400m-patch14-384
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: None
  * pretrain lr&batch size: 1e-3 256
  * finetune lr&batch size: 2e-5 128
  * data: LLaVA-1.5
  * data type: fp16
## 训练策略：
预训练：LLaVA-1.5
微调：LLaVA-1.5
## 实验结果：
  * GQA: 58.63
  * SQA: 60.24
  * TextVQA: 49.06
  * VQAv2:
  * VizWiz:
## 实验分析：
本次实验将CLIP替换成了SigLip， SigLip有729个visual tokens(分辨率为384), 似乎效果提升了？需要进一步检验其他效果。现在必须确定有哪些语言模型和视觉模型值得进一步实验，
我认为应该确立是TinyLlama, StableLM, 和Phi，但Phi一直没有训练成功，为了保证效率，应该先训练TinyLlama和StableLM的四个版本，预计需要24小时

# 实验4：stablelm-standard-data-siglip
## 实验时间：2024年2月13日10点02分
## 实验重要参数：
  * LLM: stabilityai/stablelm-2-zephyr-1_6b
  * VT: google/siglip-so400m-patch14-384
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: None
  * pretrain lr&batch size: 1e-3 256
  * finetune lr&batch size: 2e-5 128
  * data: LLaVA-1.5
  * data type: bf16
## 训练策略：
预训练：LLaVA-1.5
微调：LLaVA-1.5
## 实验结果：
  * GQA: 61.13
  * SQA: 62.77
  * TextVQA: 54.09
  * VQAv2:
  * VizWiz:
## 实验分析：

# 实验5：phi-standard-data-siglip
## 实验时间：2024年2月9日14点45分
## 实验重要参数：
  * LLM: microsoft/phi-2
  * VT: google/siglip-so400m-patch14-384
  * CM: MLP
  * LoRA: Yes
  * Unlock ViT From: None
  * pretrain lr&batch size: 1e-3 256
  * finetune lr&batch size: 2e-4 128 LoRA & 2e-5 mlp
  * data: LLaVA-1.5
  * data type: fp16
## 训练策略：
预训练：LLaVA-1.5
微调：LLaVA-1.5
## 实验结果：
  * GQA: 58.64
  * SQA: 67.13
  * TextVQA: 49.96
  * VQAv2:
  * VizWiz:
## 实验分析：

# V1.1
# 实验6：tinyllama-sharegpt4v-unlock-vit-from-12
## 实验时间：2024年2月2日20点00分
## 实验重要参数：
  * LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: 12
  * mm_mlp_pretrain: standard-llava-transformers-4.36.1's pretrain mlp
  * pretrain lr&batch size: 2e-5 256 (2 gradient accumulation steps)
  * finetune lr&batch size: 2e-5 128
  * data: sharegpt4v
  * data type: fp16
## 训练策略：
预训练：MLP使用standard-llava-transformers-4.36.1初始化，在sharegpt4v的pretrain数据上对齐
微调：与ShareGPT4V一致
## 实验结果：
  * GQA: 59.43
  * SQA: 58.80
  * TextVQA: 48.05
  * VQAv2: 75.24
  * VizWiz: 34.74
## 实验分析：
ShareGPT4V的论文声称，将ViT从第12层打开能够取得最好的效果，这个实验是对该结论的验证。


# 实验7：stablelm-sharegpt4v-unlock-vit-from-12
## 实验时间：2024年2月4日14点02分
## 实验重要参数：
  * LLM: stabilityai/stablelm-2-zephyr-1_6b
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: 12
  * pretrain lr&batch size: 2e-5 256
  * finetune lr&batch size: 2e-5 128
  * data: sharegpt4v
  * data type: bf16
## 训练策略：
预训练：与ShareGPT4V一致
微调：与ShareGPT4V一致
## 实验结果：
  * GQA: 60.26
  * SQA: 63.06
  * TextVQA: 51.6
  * VQAv2: 76.34
  * VizWiz: 36.34
## 实验分析：

# 实验8：tinyllama-sharegpt4v-unlock-vit-from-12-siglip
## 实验时间：2024年2月9日14点45分
## 实验重要参数：
  * LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  * VT: google/siglip-so400m-patch14-384
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: 12
  * pretrain lr&batch size: 1e-3 256
  * finetune lr&batch size: 2e-5 128
  * data: ShareGPT4V
  * data type: fp16
## 训练策略：
预训练：ShareGPT4V
微调：ShareGPT4V
## 实验结果：
  * GQA: 60.25
  * SQA: 60.14
  * TextVQA: 51.68
  * VQAv2:
  * VizWiz:
## 实验分析：

# 实验9：stablelm-sharegpt4v-unlock-vit-from-12-siglip
## 实验时间：2024年2月13日10点02分
## 实验重要参数：
  * LLM: stabilityai/stablelm-2-zephyr-1_6b
  * VT: google/siglip-so400m-patch14-384
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: 12
  * pretrain lr&batch size: 1e-3 256
  * finetune lr&batch size: 2e-5 128
  * data: ShareGPT4V
  * data type: bf16
## 训练策略：
预训练：ShareGPT4V
微调：ShareGPT4V
## 实验结果：
  * GQA: 61.93
  * SQA: 64.70
  * TextVQA: 56.39
  * VQAv2:
  * VizWiz:
## 实验分析：
