import torch
import torch.nn as nn
from src.partial_convolution2d import PartialConvolution2d

class PartialConvUNet(nn.Module):
    def __init__(self):
        super(PartialConvUNet, self).__init__()

        # Down Sampling (encoding)
        self.enc1 = nn.Sequential(PartialConvolution2d(3, 64, kernel_size=7, stride=2, padding=3), nn.ReLU())
        self.enc2 = nn.Sequential(PartialConvolution2d(64, 128, kernel_size=7, stride=2, padding=3), nn.ReLU())
        self.enc3 = nn.Sequential(PartialConvolution2d(128, 256, kernel_size=7, stride=2, padding=3), nn.ReLU())
        self.enc4 = nn.Sequential(PartialConvolution2d(256, 512, kernel_size=7, stride=2, padding=3), nn.ReLU())

        # Bottle neck
        self.bottleneck1 = nn.Sequential(PartialConvolution2d(512, 1024, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.bottleneck2 = nn.Sequential(PartialConvolution2d(1024, 1024, kernel_size=3, stride=1, padding=1), nn.ReLU())

        # Up Sampling (decoding)
        self.dec4 = nn.Sequential(PartialConvolution2d(1024+512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.dec3 = nn.Sequential(PartialConvolution2d(512+256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(PartialConvolution2d(256+128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(PartialConvolution2d(128+64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())

        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Ensure output is in the [0,1] range
        )

    def forward(self, x, mask):
        # Downsampling Path (Encoder)
        x1, mask1 = self.enc1[0](x, mask)
        # print(f"Mask after enc1: {mask1.shape}")
        x1 = self.enc1[1](x1)
        x2, mask2 = self.enc2[0](x1, mask1)
        # print(f"Mask after enc2: {mask2.shape}")
        x2 = self.enc2[1](x2)
        x3, mask3 = self.enc3[0](x2, mask2)
        # print(f"Mask after enc3: {mask3.shape}")
        x3 = self.enc3[1](x3)
        x4, mask4 = self.enc4[0](x3, mask3)
        # print(f"Mask after enc4: {mask4.shape}")
        x4 = self.enc4[1](x4)

        # Bottleneck
        b1, mask_b1 = self.bottleneck1[0](x4, mask4)
        # print(f"Mask after bottleneck1: {mask_b1.shape}")
        b1 = self.bottleneck1[1](b1)
        b2, mask_b2 = self.bottleneck2[0](b1, mask_b1)
        # print(f"Mask after bottleneck2: {mask_b2.shape}")
        b2 = self.bottleneck2[1](b2)

        # Upsampling Path - Ensuring Shape Consistency
        d4, mask_d4 = self.dec4[0](
            torch.cat([nn.functional.interpolate(b2, size=x4.shape[2:], mode='nearest'), x4], dim=1),
            nn.functional.interpolate(mask4, size=x4.shape[2:], mode='nearest')
        )
        # print(f"Mask after dec4: {mask_d4.shape}")
        d4 = self.dec4[1](d4)

        d3, mask_d3 = self.dec3[0](
            torch.cat([nn.functional.interpolate(d4, size=x3.shape[2:], mode='nearest'), x3], dim=1),
            nn.functional.interpolate(mask3, size=x3.shape[2:], mode='nearest')
        )
        # print(f"Mask after dec3: {mask_d3.shape}")
        d3 = self.dec3[1](d3)

        d2, mask_d2 = self.dec2[0](
            torch.cat([nn.functional.interpolate(d3, size=x2.shape[2:], mode='nearest'), x2], dim=1),
            nn.functional.interpolate(mask2, size=x2.shape[2:], mode='nearest')
        )
        # print(f"Mask after dec2: {mask_d2.shape}")
        d2 = self.dec2[1](d2)

        d1, mask_d1 = self.dec1[0](
            torch.cat([nn.functional.interpolate(d2, size=x1.shape[2:], mode='nearest'), x1], dim=1),
            nn.functional.interpolate(mask1, size=x1.shape[2:], mode='nearest')
        )
        # print(f"Mask after dec1: {mask_d1.shape}")
        d1 = self.dec1[1](d1)

        # Final Output
        output = self.final(d1)  # Output: (B, 3, H, W) → Reconstructed image


        # Ensure both output and mask match the original input size
        output = torch.nn.functional.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        mask_d1 = torch.nn.functional.interpolate(mask_d1, size=x.shape[2:], mode='nearest')

        return output, mask_d1
