import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class InpaintingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Load images and masks into RAM as tensors
        self.images = self._load_images(image_dir)
        self.masks = self._load_images(mask_dir, grayscale=True)

    def _load_images(self, directory, grayscale=False):
        images = []
        for filename in sorted(os.listdir(directory)):
            image_path = os.path.join(directory, filename)
            with Image.open(image_path) as img:
                if grayscale:
                    img = img.convert("L")  # Convert to grayscale for masks
                else:
                    img = img.convert("RGB")  # Convert to RGB for images
                
                img = np.array(img)  # Convert to NumPy array
                
                # Convert to PyTorch tensor immediately
                img_tensor = torch.tensor(img, dtype=torch.float32)

                # Ensure correct shape for PyTorch (C, H, W)
                if grayscale:
                    img_tensor = img_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
                else:
                    img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                
                images.append(img_tensor)
        
        return torch.stack(images)  # Store all images as a single tensor in RAM

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve preloaded image and mask
        image = self.images[idx]
        mask = self.masks[idx]

        # Apply transformations to the image only
        if self.transform:
            image = self.transform(image)

        # Normalize mask to binary values (1 = valid, 0 = missing)
        mask = (mask > 0).float()

        # Generate corrupted image (zero-out missing pixels)
        corrupted_image = image * mask

        return corrupted_image, mask, image  # (Input, Mask, Target)

def get_dataloader(image_dir, mask_dir, batch_size=16, transform=None):
    dataset = InpaintingDataset(image_dir, mask_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
