import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
Taken largely from https://github.com/rampage644/wavenet/blob/master/wavenet/models.py
"""

class MaskedConv2d(nn.Conv2d):
    """
    Performs masked convolution
    """
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        mask = self.premask(self.weight, mask_type)
        mask = torch.from_numpy(mask).float()

        # Register as buffer so that the params are not updated when training
        self.register_buffer('mask', mask)

    def premask(self, W, mask_type):
        f_out, f_in, f_height, f_width = W.size()
        mask = np.ones_like(W.detach().numpy()).astype('f')
        yc, xc = f_height // 2, f_width // 2
        mask[:, :, yc+1:, :] = 0.0
        mask[:, :, yc:, xc+1:] = 0.0

        for i in range(3):
            mask[self.bmask(f_out, f_in, i, i), yc, xc] = 0.0 if mask_type == 'A' else 1.0

        # G > R
        mask[self.bmask(f_out, f_in, 0, 1), yc, xc] = 0.0
        # B > R
        mask[self.bmask(f_out, f_in, 0, 2), yc, xc] = 0.0
        # B > G
        mask[self.bmask(f_out, f_in, 1, 2), yc, xc] = 0.0

        return mask

    def bmask(self, c_out, c_in, i_out, i_in):
        """
        Same pixel masking - pixel won't access next color (conv filter dim)
        """
        c_out_idx = np.expand_dims(np.arange(c_out) % 3 == i_out, 1)
        c_in_idx = np.expand_dims(np.arange(c_in) % 3 == i_in, 0)
        a1, a2 = np.broadcast_arrays(c_out_idx, c_in_idx)

        return a1 * a2

    def forward(self, x):
        # Mask the weights before convolution
        self.weight.data *= self.mask
        x = super(MaskedConv2d, self).forward(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, h):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2 * h, h, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(h, h, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(h, 2 * h, (1, 1))
        )

    def forward(self, x):
        res = self.block(x)
        assert res.shape == x.shape
        return x + res

class PixelCNN(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels, kernel_size=7, num_hidden_blocks=7):
        super(PixelCNN, self).__init__()
        self.in_ = nn.Sequential(
            MaskedConv2d('A', in_channels=in_channels, out_channels=hidden_dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=hidden_dims),
            nn.ReLU(inplace=True)
        )
        
        layer = [
            MaskedConv2d('B', in_channels=hidden_dims, out_channels=hidden_dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=hidden_dims),
            nn.ReLU(inplace=True)
        ]

        self.block = nn.Sequential(*[l for _ in range(num_hidden_blocks) for l in layer])
        self.out = nn.Conv2d(in_channels=hidden_dims, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.in_(x)
        x = self.block(x)
        x = self.out(x)

        return x

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