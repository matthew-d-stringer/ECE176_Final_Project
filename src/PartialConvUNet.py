import torch
import torch.nn as nn
from src.partial_convolution2d import PartialConvolution2d

# So there are going to be 2 paths.
# First we are going to down sample the image using partial
# convolution
# By downsampling, we increase the number of channels so that
# We can capture more complex features
# In the bottle neck, we are going to do another convolution
# This convolution will have little affect on the dimensions of the 
# data
# It exists in order to extract the higher level patterns
# Finally, we upsample the features into an image again.

class PartialConvUNet(nn.Module):
    def __init__(self):
        super(PartialConvUNet, self).__init__()

        # Down Sampling (encoding)
        self.enc1 = PartialConvolution2d(  3,  64, kernel_size=7, stride=2, padding=3)
        self.enc2 = PartialConvolution2d( 64, 128, kernel_size=7, stride=2, padding=3)
        self.enc3 = PartialConvolution2d(128, 256, kernel_size=7, stride=2, padding=3)
        self.enc4 = PartialConvolution2d(256, 512, kernel_size=7, stride=2, padding=3)

        # Bottle neck
        self.bottleneck1 = PartialConvolution2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bottleneck2 = PartialConvolution2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        # Up Sampling (decoding)
        self.dec4 = PartialConvolution2d(1024+512, 512, kernel_size=3, stride=1, padding=1)
        self.dec3 = PartialConvolution2d(512+256, 256, kernel_size=3, stride=1, padding=1)
        self.dec2 = PartialConvolution2d(256+128, 128, kernel_size=3, stride=1, padding=1)
        self.dec1 = PartialConvolution2d(128+64, 64, kernel_size=3, stride=1, padding=1)

        # Final output
        self.final = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # Pooling and Upsampling
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, mask):

        # Downsampling Path (Encoder)
        x1, mask1 = self.enc1(x, mask)  # Output: (B, 64, H/2, W/2)
        x2, mask2 = self.enc2(x1, mask1)  # Output: (B, 128, H/4, W/4)
        x3, mask3 = self.enc3(x2, mask2)  # Output: (B, 256, H/8, W/8)
        x4, mask4 = self.enc4(x3, mask3)  # Output: (B, 512, H/16, W/16)

        # Bottleneck (Extracting high-level patterns)
        b1, mask_b1 = self.bottleneck1(x4, mask4)  # Output: (B, 1024, H/16, W/16)
        b2, mask_b2 = self.bottleneck2(b1, mask_b1)  # Output: (B, 1024, H/16, W/16)

        # Upsampling Path (Decoder with Skip Connections)
        d4, mask_d4 = self.dec4(torch.cat([self.up(b2), x4], dim=1), mask4)  # Output: (B, 512, H/8, W/8)
        d3, mask_d3 = self.dec3(torch.cat([self.up(d4), x3], dim=1), mask3)  # Output: (B, 256, H/4, W/4)
        d2, mask_d2 = self.dec2(torch.cat([self.up(d3), x2], dim=1), mask2)  # Output: (B, 128, H/2, W/2)
        d1, mask_d1 = self.dec1(torch.cat([self.up(d2), x1], dim=1), mask1)  # Output: (B, 64, H, W)

        # inal Output Layer
        output = self.final(d1)  # Output: (B, 3, H, W) → Reconstructed image

        return output
