import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        
        assert mask_type in ['A', 'B'], "Unknown Mask Type"
        
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        
        self.register_buffer('mask', self.weight.data.clone())

        _, depth, height, width = self.weight.size()
        
        self.mask.fill_(1)

        if mask_type =='A':
            self.mask[:,:,height//2,width//2:] = 0
            self.mask[:,:,height//2+1:,:] = 0
        else:
            self.mask[:,:,height//2,width//2+1:] = 0
            self.mask[:,:,height//2+1:,:] = 0


    def forward(self, x):
        self.weight.data *= self.mask
        
        return super(MaskedConv2d, self).forward(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, kernel):
        super(ResidualBlock, self).__init__()
        self.layer_1 = PixelCNNLayer(in_channels=dim, out_channels=dim // 2, kernel=1)
        self.layer_2 = PixelCNNLayer(in_channels=dim // 2, out_channels=dim // 2, kernel=kernel)
        self.layer_3 = PixelCNNLayer(in_channels=dim // 2, out_channels=dim, kernel=1)

    def forward(self, x):
        res = self.layer_1(x)
        res = self.layer_2(res)
        res = self.layer_3(res)

        assert res.shape == x.shape

        return x + res

class PixelCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(PixelCNNLayer, self).__init__()

        self.conv = MaskedConv2d('B', in_channels, out_channels, kernel, 1, kernel // 2, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x

class PixelCNN(nn.Module):
    """
    Network of PixelCNN as described in A Oord et. al. 
    """
    def __init__(self, kernel=7, channels=128, last_layer_filters=256, device=None):
        super(PixelCNN, self).__init__()
        self.kernel = kernel
        self.channels = channels
        self.layers = {}
        self.device = device

        self.Conv2d_1 = MaskedConv2d('A',in_channels=1,out_channels=channels, kernel_size=kernel, stride=1, padding=kernel//2, bias=False)
        self.BatchNorm2d_1 = nn.BatchNorm2d(channels)
        self.ReLU_1= nn.ReLU(True)

        self.res1 = ResidualBlock(channels, kernel=3)
        self.res2 = ResidualBlock(channels, kernel=3)
        self.res3 = ResidualBlock(channels, kernel=3)
        self.res4 = ResidualBlock(channels, kernel=3)
        self.res5 = ResidualBlock(channels, kernel=3)
        self.res6 = ResidualBlock(channels, kernel=3)
        self.res7 = ResidualBlock(channels, kernel=3)
        self.res8 = ResidualBlock(channels, kernel=3)
        self.res9 = ResidualBlock(channels, kernel=3)
        self.res10 = ResidualBlock(channels, kernel=3)
        self.res11 = ResidualBlock(channels, kernel=3)
        self.res12 = ResidualBlock(channels, kernel=3)
        self.res13 = ResidualBlock(channels, kernel=3)
        self.res14 = ResidualBlock(channels, kernel=3)
        self.res15 = ResidualBlock(channels, kernel=3)

        self.layer_2 = PixelCNNLayer(in_channels=channels, out_channels=last_layer_filters, kernel=1)
        self.layer_3 = PixelCNNLayer(in_channels=last_layer_filters, out_channels=last_layer_filters, kernel=1)

        self.out = nn.Conv2d(last_layer_filters, 256, 1)

    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.BatchNorm2d_1(x)
        x = self.ReLU_1(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        x = self.res10(x)
        x = self.res11(x)
        x = self.res12(x)
        x = self.res13(x)
        x = self.res14(x)
        x = self.res15(x)

        x = self.layer_2(x)
        x = self.layer_3(x)

        return self.out(x)

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