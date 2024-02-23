# 实验1：unlock-vit-from-12-tune-entire-model
## 实验时间：2024年1月30日23点10分
## 实验重要参数：
  * LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: 12
  * pretrain lr&batch size: 2e-5 256
  * finetune lr&batch size: 2e-5 128
  * data: standard llava-1.5
  * data type: fp16
## 训练策略：
预训练：将ViT, MLP和LLM同时打开，跟随ShareGPT4V的论文，ViT从第12层开始打开
微调：与LLaVA一致
## 实验结果：
  * GQA: 58.28
  * SQA: 57.06
  * TextVQA: 43.17
  * VQAv2: 74.02
  * VizWiz:
  * MMVet:
  * POPE: adversarial: 0.835 random: 0.876 popular: 0.869
## 实验分析：
本次实验中TextVQA和baseline(46.37)的效果差很多，我认为有可能是因为微调CLIP使CLIP的泛化性受到损伤，而TextVQA这个任务是非常细粒度的任务，导致效果减少最大。如果要提升效果，应当从更好的数据（ShareGPT4V尝试）

# 实验2：unlock-vit-from-18-tune-entire-model
## 实验时间：2024年1月31日13点37分
## 实验重要参数：
  * LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: 18
  * pretrain lr&batch size: 2e-5 256
  * finetune lr&batch size: 2e-5 128
  * data: standard llava-1.5
  * data type: fp16
## 训练策略：
预训练：将ViT, MLP和LLM同时打开，ViT从第18层开始打开
微调：与LLaVA一致
## 实验结果：
  * GQA: 58.32
  * SQA: 54.24
  * TextVQA: 43.44
  * VQAv2: 73.89
  * VizWiz:
  * POPE: adversarial: 0.840 random: 0.876 popular: 0.870
## 实验分析：
本次实验中TextVQA和SQA与baseline(46.37， 59.4)的效果差很多，和上组实验的分析相同，应该是实验数据对CLIP的泛化性损伤了。

# 实验3：unlock-vit-from-21-tune-entire-model
## 实验时间：2024年1月30日13点37分
## 实验重要参数：
  * LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: 21
  * pretrain lr&batch size: 2e-5 256
  * finetune lr&batch size: 2e-5 128
  * data: standard llava-1.5
  * data type: fp16
## 训练策略：
预训练：将ViT, MLP和LLM同时打开，ViT从第21层开始打开
微调：与LLaVA一致
## 实验结果：
  * GQA: 58.17
  * SQA: 58.25
  * TextVQA: 43.93
  * VQAv2:
  * VizWiz:
  * POPE: adversarial: 0.838 random: 0.875 popular: 0.867
## 实验分析：
SQA的表现与从18打开相比略好，但12, 18, 21之间没有观察到可见规律，需要看看第15层打开时什么情况

# 实验4：standard-llava-transformers-4.36.1
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

# 实验5：sharegpt4v-unlock-vit-from-18-tune-entire-model
## 实验时间：2024年2月2日10点40分
## 实验重要参数：
  * LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: 18
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
  * SQA: 58.7
  * TextVQA: 48.22
  * VQAv2:
  * VizWiz:
## 实验分析：
总是报OOM的错，不得不把gradient accumulation step调为2。实验时间很长全部完成大约需要6至7小时。忘记上传sharetext_vqa数据集，现已上传。
这次实验在总体上效果有一些提升，特别是TextVQA部分，提升显著，这与训练数据有强相关性，这次实验说明，想要训练更好的模型，不光要从参数量考虑，也要从数据质量考虑。但是让我感到奇怪的是，为什么在预训练过程中loss会降到这么低(低于0.5)？需要进一步探究。

# 实验6：sharegpt4v-unlock-vit-from-12-tune-entire-model
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
  * MMVet: 25.1
  * POPE: adversarial: 0.839 random: 0.880 popular: 0.858
## 实验分析：
ShareGPT4V的论文声称，将ViT从第12层打开能够取得最好的效果，这个实验是对该结论的验证。

# 实验7：sharegpt4v-unlock-vit-from-15-tune-entire-model
## 实验时间：2024年2月2日20点00分
## 实验重要参数：
  * LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: 15
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
  * SQA: 58.95
  * TextVQA: 48.18
  * VQAv2: 75.23
  * VizWiz:
  * MMVet: 24
  * POPE: adversarial: 0.840 random: 0.880 popular: 0.860
## 实验分析：
消融实验，与从第12层，18层组成一组消融实验，有时间还可以做21层

# 实验8：moe-mlp-unlock-vit-from-12-tune-entire-model
## 实验时间：2024年2月2日20点00分
## 实验重要参数：
  * LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: 12
  * pretrain: 使用sharegpt4v-unlock-vit-from-12-tune-entire-model作为初始化
  * finetune lr&batch size: 2e-5 128
  * data: sharegpt4v
  * data type: fp16
## 训练策略：
微调：与ShareGPT4V一致
## 实验结果：
  * GQA:
  * SQA:
  * TextVQA:
  * VQAv2:
  * VizWiz:
## 实验分析：
FAILED


# 实验9：stablelm-standard-data
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
  * MMVet: 25.0
  * POPE: adversarial: 0.840 random: 0.872 popular: 0.863
## 实验分析：


# 实验10：stablelm-sharegpt4v-unlock-vit-from-12
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
  * MMVet: 29.3
  * POPE: adversarial: 0.844 random: 0.864 popular: 0.855
## 实验分析：


# 实验11：stablelm-sharegpt4v
## 实验时间：2024年2月6日16点30分
## 实验重要参数：
  * LLM: stabilityai/stablelm-2-zephyr-1_6b
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: No
  * pretrain lr&batch size: 1e-3 256
  * finetune lr&batch size: 2e-5 128
  * data: sharegpt4v
  * data type: bf16
## 训练策略：
预训练：与LLaVA-1.5一致
微调：与ShareGPT4V一致
## 实验结果：
  * GQA: 59.67
  * SQA: 63.41
  * TextVQA: 50.38
  * VQAv2: 75.89
  * VizWiz:
  * MMVet: 27.4
  * POPE: adversarial: 0.847 random: 0.878 popular: 0.869
## 实验分析：


# 实验12：minicpm-standard-data
## 实验时间：2024年2月9日14点45分
## 实验重要参数：
  * LLM: openbmb/MiniCPM-2B-dpo-bf16
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: No
  * pretrain lr&batch size: 1e-3 256
  * finetune lr&batch size: 2e-5 128
  * data: standard-data
  * data type: bf16
## 训练策略：
预训练：与LLaVA-1.5一致
微调：与LLaVA-1.5一致
## 实验结果：
  * GQA:
  * SQA:
  * TextVQA:
  * VQAv2:
  * VizWiz:
## 实验分析：
FAILED

# 实验13：minicpm-sharegpt4v-unlock-vit-from-12
## 实验时间：2024年2月9日14点45分
## 实验重要参数：
  * LLM: openbmb/MiniCPM-2B-dpo-bf16
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: 12
  * pretrain lr&batch size: 1e-3 256
  * finetune lr&batch size: 2e-5 128
  * data: sharegpt4v
  * data type: bf16
## 训练策略：
预训练：与ShareGPT4V
微调：与ShareGPT4V
## 实验结果：
  * GQA:
  * SQA:
  * TextVQA:
  * VQAv2:
  * VizWiz:
## 实验分析：
FAILED

# 实验14：tinyllama-standard-data-siglip
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
  * VQAv2: 75.8
  * VizWiz:
  * MMVet: 24.1
  * POPE: adversarial: 0.847 random: 0.875 popular: 0.862
## 实验分析：
本次实验将CLIP替换成了SigLip， SigLip有729个visual tokens(分辨率为384), 似乎效果提升了？需要进一步检验其他效果。现在必须确定有哪些语言模型和视觉模型值得进一步实验，
我认为应该确立是TinyLlama, StableLM, 和Phi，但Phi一直没有训练成功，为了保证效率，应该先训练TinyLlama和StableLM的四个版本，预计需要24小时

# 实验15：phi-standard-data-siglip-lora
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
  * MMVet:
  * POPE:
## 实验分析：


# 实验16：stablelm-standard-data-siglip
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
  * VQAv2: 78.14
  * VizWiz
  * MMVet: 29.5
  * POPE: adversarial: 0.853 random: 0.880 popular: 0.874
## 实验分析：


# 实验17：stablelm-sharegpt4v-unlock-vit-from-12-siglip
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
  * VQAv2: 78.91
  * VizWiz:
  * MMVet: 32.6
  * POPE: adversarial: 0.851 random: 0.878 popular: 0.867
## 实验分析：


# 实验18：tinyllama-sharegpt4v-unlock-vit-from-12-siglip
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
  * VQAv2: 76.89
  * VizWiz:
  * MMVet: 25.8
  * POPE: adversarial: 0.847 random: 0.875 popular: 0.862
## 实验分析：

# 实验19：phi-standard-data-siglip
## 实验时间：2024年2月15日14点45分
## 实验重要参数：
  * LLM: microsoft/phi-2
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
  * GQA: 61.34
  * SQA: 69.91
  * TextVQA: 55.64
  * VQAv2: 79.2
  * VizWiz: 38.45
  * MMVet: 32.1
  * POPE: adversarial: 0.857 random: 0.885 popular: 0.871
  * LLaVAW: 67.9
## 实验分析：

# 实验20：phi-sharegpt4v-unlock-vit-from-12-siglip
## 实验时间：2024年2月16日14点45分
## 实验重要参数：
  * LLM: microsoft/phi-2
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
  * GQA: 61.97
  * SQA: 69.06
  * TextVQA: 59.13
  * VQAv2: 79.93
  * VizWiz: 34.42? weird
  * MMVet: 32.0
  * POPE: adversarial: 0.856 random: 0.873 popular: 0.863
  * LLaVAW: 75.8
## 实验分析：

# 实验21：phi-standard-data
## 实验时间：2024年2月17日14点45分
## 实验重要参数：
  * LLM: microsoft/phi-2
  * VT: openai/clip-vit-large-patch14-336
  * CM: MLP
  * LoRA: No
  * Unlock ViT From: No
  * pretrain lr&batch size: 1e-3 256
  * finetune lr&batch size: 2e-5 128
  * data: LLaVA-1.5
  * data type: fp16
## 训练策略：
预训练：LLaVA-1.5
微调：LLaVA-1.5
## 实验结果：
  * GQA:
  * SQA:
  * TextVQA:
  * VQAv2:
  * VizWiz:
## 实验分析：


# 实验22：phi-sharegpt4v-unlock-vit-from-12
## 实验时间：2024年2月17日14点45分
## 实验重要参数：
  * LLM: microsoft/phi-2
  * VT: openai/clip-vit-large-patch14-336
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
  * GQA:
  * SQA:
  * TextVQA:
  * VQAv2:
  * VizWiz:
## 实验分析：


