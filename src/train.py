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

def train(model, train_dataloader, val_dataloader, optimizer, device, num_epochs=10, patience=3):
    model.train()
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        epoch_start_time = time.time()  # Track the start time of the epoch
        print(f"Epoch {epoch+1}/{num_epochs} started.")
        
        # Use tqdm for progress bar
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
            for batch_idx, (corrupted, mask, target) in pbar:
                batch_start_time = time.time()
                
                corrupted, mask, target = corrupted.to(device), mask.to(device), target.to(device)
                optimizer.zero_grad()
                inpainted_output, _ = model(corrupted, mask)
                loss = compute_loss(inpainted_output, target, mask)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                batch_time = time.time() - batch_start_time
                remaining_time = batch_time * (len(train_dataloader) - batch_idx - 1)
                pbar.set_postfix({'Remaining Time': f'{remaining_time:.2f}s', 'Loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_dataloader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation Loss Calculation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for corrupted, mask, target in val_dataloader:
                corrupted, mask, target = corrupted.to(device), mask.to(device), target.to(device)
                inpainted_output, _ = model(corrupted, mask)
                loss = compute_loss(inpainted_output, target, mask)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save model checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"checkpoints/inpainting_model_epoch{epoch+1}_{timestamp}_valLoss{avg_val_loss:.4f}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")
        
        # Early Stopping Logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break
        
        model.train()  # Switch back to training mode

def main():
    image_dir_train = "dataset/images/training_resized"
    mask_dir_train = "dataset/masks/training_resized"
    image_dir_val = "dataset/images/validation_resized"
    mask_dir_val = "dataset/masks/validation_resized"

    print("Loading training data...")
    train_dataloader = get_dataloader(image_dir_train, mask_dir_train, batch_size=32, transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
    print(f"Training data loaded. Total batches: {len(train_dataloader)}")

    print("Loading validation data...")
    val_dataloader = get_dataloader(image_dir_val, mask_dir_val, batch_size=32, transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
    print(f"Validation data loaded. Total batches: {len(val_dataloader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartialConvUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train(model, train_dataloader, val_dataloader, optimizer, device, num_epochs=50, patience=5)

    writer.close()

if __name__ == "__main__":
    main()
