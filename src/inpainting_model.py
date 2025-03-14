import torch
import torch.nn as nn
from src.partial_convolution2d import PartialConvolution2d

class InpaintingModel(nn.Module):
    def __init__(self):
        super(InpaintingModel, self).__init__()

        self.conv1 = PartialConvolution2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = PartialConvolution2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = PartialConvolution2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 3, kernel_size=3, padding=1)  # Output image (RGB)

    def forward(self, x, mask):
        x, mask = self.conv1(x, mask)
        x, mask = self.conv2(x, mask)
        x, mask = self.conv3(x, mask)
        x = self.conv4(x)
        return x
