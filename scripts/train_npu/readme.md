# NPU训练指南

## 安装环境

torch_npu的版本务必于CANN一致

## 训练

1. deepspeed的配置文件使用`scripts/zero3_npu.json`, 重点是将`zero_optimization`中的`'auto'`改为具体的值
2. 设置`attn_implementation`的值为 `sdpa`而非`flash_attention_2 `
3. 注意修改使用的显卡ID, 输出位置, 数据集位置等
