import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class InpaintingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask on demand (not preloading)
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # Grayscale for masks

        # Convert to PyTorch tensors with efficient transforms
        image = transforms.ToTensor()(image)  # Normalizes to [0,1] and (C, H, W)
        mask = transforms.ToTensor()(mask)    # Normalizes to [0,1] and (1, H, W)

        # Normalize mask to binary values (1 = valid, 0 = missing)
        mask = (mask > 0).float()

        # Generate corrupted image (zero-out missing pixels)
        corrupted_image = image * mask

        return corrupted_image, mask, image  # (Input, Mask, Target)

def get_dataloader(image_dir, mask_dir, batch_size=16, transform=None, num_workers=4):
    dataset = InpaintingDataset(image_dir, mask_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader
