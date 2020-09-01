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
    res = modules.ResidualBlock(3, 3)

def test_PixelCNN_is_valid():
    pixel_cnn = modules.PixelCNN(in_channels=1, channels=256, kernel=7, last_layer_filters=256)

def test_CroppedConv2d():
    x = np.random.normal(size=(1, 1, 9, 9))
    t = torch.from_numpy(x).double()

    c = modules.CroppedConv2d(in_channels=1, out_channels=1, kernel_size=3).double()
    c.forward(t)