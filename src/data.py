import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class InpaintingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(os.path.join(self.image_dir, self.image_paths[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.mask_paths[idx])).convert("L")  # Load as grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Normalize mask to binary values (1 = valid, 0 = missing)
        mask = (mask > 0).float()

        # Generate corrupted image (zero-out missing pixels)
        corrupted_image = image * mask

        return corrupted_image, mask, image  # (Input, Mask, Target)

def get_dataloader(image_dir, mask_dir, batch_size=16, transform=None):
    dataset = InpaintingDataset(image_dir, mask_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
