import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from data import get_dataloader
from PartialConvUNet import PartialConvUNet
from loss_func import compute_loss
from datetime import datetime
from tqdm import tqdm
from PIL import Image

# Load trained model
def load_model(checkpoint_path, device):
    model = PartialConvUNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def save_inpainted_images(original, masked, inpainted, output_folder, filename):
    """ Save inpainted images for analysis """
    os.makedirs(output_folder, exist_ok=True)

    original = (original.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    masked = (masked.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    inpainted = (inpainted.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_folder, f"{filename}_original.jpg"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_folder, f"{filename}_masked.jpg"), cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_folder, f"{filename}_inpainted.jpg"), cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))

def validate(model, dataloader, device, output_folder="validation_results"):
    model.eval()
    total_loss = 0

    os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", unit="batch")
        
        for batch_idx, (corrupted, mask, target) in enumerate(progress_bar):
            corrupted, mask, target = corrupted.to(device), mask.to(device), target.to(device)
            
            # Perform inpainting
            inpainted_output = model(corrupted * mask, mask)

            # Compute loss
            loss = compute_loss(inpainted_output, target, mask)
            total_loss += loss.item()

            # Save a few sample images
            if batch_idx < 10:  # Save first 10 samples for visualization
                save_inpainted_images(target[0].cpu(), corrupted[0].cpu(), inpainted_output[0].cpu(), output_folder, f"sample_{batch_idx}")

        avg_loss = total_loss / len(dataloader)
        print(f"Validation Complete. Average Loss: {avg_loss:.4f}")

def main():
    # Paths
    validation_image_dir = "dataset/images/validation_resized"
    validation_mask_dir = "dataset/masks/validation_resized"

    # Parameters
    batch_size = 32
    checkpoint_path = "checkpoints/inpainting_model_20250315_074526.pth"  # Update as needed

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load validation data
    print("Loading validation data...")
    dataloader = get_dataloader(validation_image_dir, validation_mask_dir, batch_size=batch_size, transform=transform)
    print(f"Validation data loaded. Total batches: {len(dataloader)}")

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(checkpoint_path, device)

    # Validate
    print("Starting validation...")
    validate(model, dataloader, device)

if __name__ == "__main__":
    main()
