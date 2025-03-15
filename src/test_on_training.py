import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data import get_dataloader
from datetime import datetime
from PartialConvUNet import PartialConvUNet
from loss_func import compute_loss

def run_model_on_training_data(model_path, image_dir, mask_dir, batch_size=32):
    """Loads a trained model, runs it on the training dataset, and computes loss."""
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Load data
    print("Loading data...")
    dataloader = get_dataloader(image_dir, mask_dir, batch_size=batch_size, transform=transform)
    print(f"Data loaded. Total batches: {len(dataloader)}")
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = PartialConvUNet().to(device)
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    
    total_loss = 0
    
    # Run the model on the training data
    with torch.no_grad():
        for batch_idx, (corrupted, mask, target) in enumerate(dataloader):
            print(f"Processing batch {batch_idx+1}/{len(dataloader)}...")
            
            corrupted, mask, target = corrupted.to(device), mask.to(device), target.to(device)
            
            # Forward pass
            inpainted_output = model(corrupted, mask)
            
            # Compute loss
            loss = compute_loss(inpainted_output, target, mask)
            total_loss += loss.item()
            
            # Display some results (for debugging/analysis)
            print(f"Batch {batch_idx+1} processed. Loss: {loss.item():.4f}")
            
    avg_loss = total_loss / len(dataloader)
    print(f"Processing complete. Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    model_checkpoint = "checkpoints/inpainting_model_20250315_071514.pth"  # Change this if needed
    image_dir = "dataset/images/training_resized"
    mask_dir = "dataset/masks/training_resized"
    run_model_on_training_data(model_checkpoint, image_dir, mask_dir)
