import modules
import numpy as np
import torch

def test_is_gpu_available():
    assert torch.cuda.is_available()

def test_mask_A_zeros_after_center():
    W = np.zeros((1, 1, 7, 7), dtype=float)
    mask = modules.mask_A(W)

    for i in range(4):
        assert np.where(mask[:, :, 3:, 4:] > 0).count(i) == 0
        assert mask[0, 0, 3, 3] == 1

def test_casualconv1d_valid():
    inp = torch.from_numpy(np.ones((3,2,5))).float()
    CaConv1d = modules.CausalConv1d(in_channels=2, out_channels=6, kernel_size=2, dilation=1)
    out = CaConv1d(inp)
