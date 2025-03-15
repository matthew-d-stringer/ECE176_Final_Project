import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data import get_dataloader
from datetime import datetime
from PartialConvUNet import PartialConvUNet

from loss_func import compute_loss

def train(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Epoch {epoch+1}/{num_epochs} started.")
        
        for batch_idx, (corrupted, mask, target) in enumerate(dataloader):
            print(f"  Processing batch {batch_idx+1}/{len(dataloader)}...")
            
            # Move data to device
            corrupted, mask, target = corrupted.to(device), mask.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            print(f"    Performing forward pass...")
            inpainted_output = model(corrupted, mask)
            
            # Compute loss
            print(f"    Computing loss...")
            loss = compute_loss(inpainted_output, target, mask)
            
            # Backward pass
            print(f"    Backpropagating loss...")
            loss.backward()
            
            # Optimizer step
            print(f"    Updating model weights...")
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss / len(dataloader):.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} finished.")

def main():
    # Paths
    image_dir = "dataset/images"
    mask_dir = "dataset/masks"

    # Training parameters
    batch_size = 32
    num_epochs = 2
    lr = 1e-2

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
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    print("Starting training...")
    train(model, dataloader, optimizer, device, num_epochs)

    # Ensure the checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"checkpoints/inpainting_model_{timestamp}.pth"

    # Save trained model
    print(f"Saving model to {checkpoint_path}...")
    torch.save(model.state_dict(), checkpoint_path)
    print("Model saved.")

if __name__ == "__main__":
    main()
