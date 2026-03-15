import torch

try:
    import torch_npu

    has_gpu = False


    # option = {"ACL_PRECISION_MODE": "must_keep_origin_dtype"}
    # torch_npu.npu.set_option(option)

    def get_device():
        return torch.device('npu')
except ImportError:
    has_gpu = torch.cuda.is_available()


    def get_device():
        return torch.device("cuda")


def test():
    device = get_device()
    x = torch.randn(1, 3, 224, 224)
    print(x.shape, x.dtype, x.device)
    x = x.to(device)
    print(x.shape, x.dtype, x.device)
