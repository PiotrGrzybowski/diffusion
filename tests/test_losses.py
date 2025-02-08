import torch


def test_compare_implementations():
    x = torch.randn(1, 1, 28, 28)
    means = x.mean(dim=(2, 3), keepdim=True)
    variances = x.var(dim=(2, 3), unbiased=False, keepdim=True)
    log_scales = 0.5 * torch.log(variances + 1e-8)

    # print(x.shape, means.shape, variances.shape, log_scales.shape)
