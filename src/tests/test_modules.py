import modules
import numpy as np
import torch

def test_casualconv1d_valid():
    inp = torch.from_numpy(np.ones((3,2,5))).float()
    CaConv1d = modules.CausalConv1d(in_channels=2, out_channels=6, kernel_size=2, dilation=1)
    out = CaConv1d(inp)
