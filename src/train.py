import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data import get_dataloader
from datetime import datetime
from PartialConvUNet import PartialConvUNet
from torch.utils.tensorboard import SummaryWriter
from loss_func import compute_loss
from tqdm import tqdm
import time

# Set up TensorBoard writer
writer = SummaryWriter(log_dir="logs")

def train(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        epoch_start_time = time.time()  # Track the start time of the epoch
        print(f"Epoch {epoch+1}/{num_epochs} started.")
        
        # Use tqdm for progress bar
        with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
            for batch_idx, (corrupted, mask, target) in pbar:
                # Track batch start time
                batch_start_time = time.time()

                corrupted, mask, target = corrupted.to(device), mask.to(device), target.to(device)
                optimizer.zero_grad()
                inpainted_output = model(corrupted, mask)
                loss = compute_loss(inpainted_output, target, mask)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Update tqdm with the estimated time left for this epoch
                batch_time = time.time() - batch_start_time
                elapsed_time = time.time() - epoch_start_time
                remaining_time = batch_time * (len(dataloader) - batch_idx - 1)  # Estimate remaining time

                # Update progress bar with postfix
                pbar.set_postfix({'Remaining Time': f'{remaining_time:.2f}s', 'Loss': loss.item()})

        # After each epoch, log the total loss to TensorBoard
        writer.add_scalar('Loss/epoch', total_loss / len(dataloader), epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss / len(dataloader):.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} finished.")

        # Generate a timestamp for saving the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"checkpoints/inpainting_model_epoch{epoch+1}_{timestamp}_loss{total_loss/len(dataloader):.4f}.pth"
        
        # Save the model after each epoch with epoch number and loss
        print(f"Saving model for epoch {epoch+1} to {model_save_path}...")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved for epoch {epoch+1}.")

def main():
    image_dir_train = "dataset/images/training_resized"
    mask_dir_train = "dataset/masks/training_resized"
    image_dir_val = "dataset/images/validation_resized"
    mask_dir_val = "dataset/masks/validation_resized"

    # Load data
    print("Loading training data...")
    train_dataloader = get_dataloader(image_dir_train, mask_dir_train, batch_size=32, transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
    print(f"Training data loaded. Total batches: {len(train_dataloader)}")

    print("Loading validation data...")
    val_dataloader = get_dataloader(image_dir_val, mask_dir_val, batch_size=32, transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
    print(f"Validation data loaded. Total batches: {len(val_dataloader)}")

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartialConvUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train(model, train_dataloader, optimizer, device, num_epochs=5)

    # Save the model
    os.makedirs("checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"checkpoints/inpainting_model_{timestamp}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
