import modules
import numpy as np
import torch
import pytest

@pytest.fixture
def W():
    return np.zeros((1, 1, 7, 7), dtype=float)

def test_is_gpu_available():
    assert torch.cuda.is_available()

def test_casualconv1d_valid():
    inp = torch.from_numpy(np.ones((3,2,5))).float()
    CaConv1d = modules.CausalConv1d(in_channels=2, out_channels=6, kernel_size=2, dilation=1)
    out = CaConv1d(inp)

def test_masked_conv2d_init(W):
    masked_conv2d_A = modules.MaskedConv2d('A', in_channels=3, out_channels=3, kernel_size=(1, 1))
    masked_conv2d_B = modules.MaskedConv2d('B', in_channels=3, out_channels=3, kernel_size=(1, 1))

def test_residual_init():
    res = modules.ResidualBlock(3)