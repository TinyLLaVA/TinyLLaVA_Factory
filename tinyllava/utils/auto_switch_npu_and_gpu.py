import torch

try:
    import torch_npu


    def get_device():
        return torch.device('npu')
except ImportError:

    def get_device():
        return torch.device("cuda")


def test():
    device = get_device()
    x = torch.randn(1, 3, 224, 224)
    print(x.shape, x.dtype, x.device)
    x = x.to(device)
    print(x.shape, x.dtype, x.device)
