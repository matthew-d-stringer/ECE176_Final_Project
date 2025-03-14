import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

def masked_l1_loss(output, target, mask):
    """ Compute L1 loss only on masked (missing) pixels """
    return torch.mean(torch.abs((1 - mask) * (output - target)))

def total_variation_loss(img):
    """ Smoothness loss (TV loss) to remove checkerboard artifacts """
    loss = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + \
           torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    return loss

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device="cuda"):
        super(VGGPerceptualLoss, self).__init__()
        
        # Load VGG-16 Model with pretrained weights
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()

        # Move the model to the correct device (CPU or GPU)
        self.vgg.to(device)

        # Freeze parameters (we do not train VGG)
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.device = device

    def forward(self, output, target):
        """ Compute perceptual loss between output and target """
        
        # Ensure tensors are on the same device as VGG
        output = output.to(self.device)
        target = target.to(self.device)
        
        vgg_out = self.vgg(output)
        vgg_target = self.vgg(target)
        
        return F.l1_loss(vgg_out, vgg_target)

# Example usage in training loop
def compute_loss(output, target, mask):
    hole_loss = masked_l1_loss(output, target, mask)
    valid_loss = masked_l1_loss(output, target, 1 - mask)  # Inverted mask for valid pixels
    tv_loss = total_variation_loss(output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceptual_loss_fn = VGGPerceptualLoss(device)  # Create loss function with correct device

    perceptual_loss = perceptual_loss_fn(output, target)  # Now everything is on the same device

    # Weighted sum from paper
    total_loss = valid_loss + 6 * hole_loss + 0.05 * perceptual_loss + 0.1 * tv_loss
    return total_loss
