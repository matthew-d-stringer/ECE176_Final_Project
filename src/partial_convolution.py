import torch
import torch.nn as nn

class PartialConvolution2d(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels,
            kernal_size,
            stride=1,
            padding=0,
            bias=True
        ):
        super(PartialConvolution2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding, bias=bias)

        self.mask_conv = nn.Conv2d(1,1, kernal_size, stride, padding, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0)
        self.mask_conv.requires_grad_ = False

    def forward(self, x, mask):
        x_masked = x * mask # Apply mask (so that only pixels that have been learned are used)

        output = self.conv(x_masked)

        with torch.no_grad():
            mask_sum = self.mask_conv(mask)
            mask_sum = torch.clamp(mask_sum, min=1e-8)
        
        output = output/mask_sum
        new_mask = torch.where(mask_sum > 0, torch.ones_like(mask), torch.zeros_like(mask))

        return output, new_mask
