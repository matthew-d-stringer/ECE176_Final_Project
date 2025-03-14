import torch
import torch.nn.functional as F
from torchvision.models import vgg16

def masked_l1_loss(output, target, mask):
    """ Compute L1 loss only on masked (missing) pixels """
    return torch.mean(torch.abs((1 - mask) * (output - target)))

def total_variation_loss(img):
    """ Smoothness loss (TV loss) to remove checkerboard artifacts """
    loss = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + \
           torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    return loss

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16].eval()  # Use first few layers
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, output, target):
        """ Compute perceptual loss between output and target """
        vgg_out = self.vgg(output)
        vgg_target = self.vgg(target)
        return F.l1_loss(vgg_out, vgg_target)

# Example usage in training loop
def compute_loss(output, target, mask):
    hole_loss = masked_l1_loss(output, target, mask)
    valid_loss = masked_l1_loss(output, target, 1 - mask)  # Inverted mask for valid pixels
    tv_loss = total_variation_loss(output)
    perceptual_loss = VGGPerceptualLoss()(output, target)

    # Weighted sum from paper
    total_loss = valid_loss + 6 * hole_loss + 0.05 * perceptual_loss + 0.1 * tv_loss
    return total_loss
