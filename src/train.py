import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data import get_dataloader
from inpainting_model import InpaintingModel

from loss_func import compute_loss

def train(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for corrupted, mask, target in dataloader:
            corrupted, mask, target = corrupted.to(device), mask.to(device), target.to(device)
            
            optimizer.zero_grad()
            inpainted_output = model(corrupted, mask)  # Forward pass
            
            loss = compute_loss(inpainted_output, target, mask)  # Compute loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

def main():
    # Paths
    image_dir = "dataset/images"
    mask_dir = "dataset/masks"

    # Training parameters
    batch_size = 16
    num_epochs = 10
    lr = 1e-4

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load data
    dataloader = get_dataloader(image_dir, mask_dir, batch_size=batch_size, transform=transform)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InpaintingModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    train(model, dataloader, optimizer, device, num_epochs)

    # Save trained model
    torch.save(model.state_dict(), "checkpoints/inpainting_model.pth")

    print("Model saved successfully!")    

if __name__ == "__main__":
    main()
