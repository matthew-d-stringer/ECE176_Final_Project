import torch
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.PartialConvUNet import PartialConvUNet  # Ensure this is your trained model class
from PIL import Image
from numpy.random import default_rng

# Load trained model
MODEL_PATH = "checkpoints/inpainting_model_20250315_074526.pth"
# MODEL_PATH = "checkpoints/inpainting_model_20250315_071514.pth"
# MODEL_PATH = "checkpoints/inpainting_model_20250315_074526.pth"
# MODEL_PATH = "checkpoints/inpainting_model_20250315_094841.pth"
# MODEL_PATH = "checkpoints/inpainting_model_epoch1_20250315_083458_loss0.6210.pth"
# MODEL_PATH = "checkpoints/inpainting_model_epoch10_20250315_212214_loss0.5248.pth"
# MODEL_PATH = "checkpoints/inpainting_model_epoch20_20250315_232820_loss0.5132.pth" 
# MODEL_PATH = "checkpoints/inpainting_model_epoch28_20250316_011003_loss0.5085.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PartialConvUNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def generate_blob_mask(img, sigma=15, threshold=175):
    height, width = img.shape[:2]

    rng = default_rng()

    noise = rng.integers(0, 255, (height, width), np.uint8, True)
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)
    
    thresh = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

cap = cv2.VideoCapture(0)

def apply_mask(frame, mask):
    """ Apply the selected mask to the image. """
    return frame * mask[:, :, np.newaxis]

def process_frame(frame):
    """ Converts frame to tensor, applies model, and returns inpainted output. """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    image_pil = Image.fromarray(image)

    mask = generate_blob_mask(np.array(image_pil))

    mask = cv2.resize(mask, (256, 256))

    image_tensor = transform(image_pil).unsqueeze(0).to(device) 
    mask_tensor = torch.tensor(mask / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # inpainting
    with torch.no_grad():
        inpainted_tensor = model(image_tensor * mask_tensor, mask_tensor)

    # Convert to NumPy
    inpainted_image = inpainted_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    inpainted_image = np.clip(inpainted_image, 0, 1)

    return image, mask, inpainted_image

def show_result(image, mask, inpainted):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    axes[2].imshow(inpainted)
    axes[2].set_title("Inpainted Output")
    
    for ax in axes:
        ax.axis("off")
    plt.show()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = cv2.resize(frame, (256, 256))
    mask = generate_blob_mask(display_frame)
    masked_frame = apply_mask(display_frame, mask)

    cv2.imshow("Live Inpainting (Press 'C' to capture, 'Q' to quit)", masked_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        print("Capturing and processing frame...")
        image, mask, inpainted = process_frame(frame)
        show_result(image, mask, inpainted)

    elif key == ord('q'):  # Quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()