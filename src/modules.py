import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mask(W):
    filters_output, filters_input, filter_height, filter_width = W.shape
    mask = np.ones_like(W).astype('f')
    yc, xc = filter_height // 2, filter_width // 2
    mask[:, :, yc+1:, :] = 0.0
    mask[:, :, yc:, xc+1:] = 0.0

    return mask

class PixelCNN():
    pass

class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True
    ):
        self.__padding__ = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding__,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)

        if self.__padding__ != 0:
            return result[:, :, :-self.__padding__]
        
        return result