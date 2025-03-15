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
# MODEL_PATH = "checkpoints/inpainting_model_epoch1_20250315_083458_loss0.6210.pth"
# MODEL_PATH = "checkpoints/inpainting_model_epoch1_20250315_083458_loss0.6210.pth"
MODEL_PATH = "checkpoints/inpainting_model_20250315_094841.pth"

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

    # Generate a unique seed based on the filename (or use the current time)
    rng = default_rng()

    noise = rng.integers(0, 255, (height, width), np.uint8, True)
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)
    
    # **Use THRESH_BINARY_INV to make blobs black and background white**
    thresh = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

# Initialize webcam
cap = cv2.VideoCapture(0)

def apply_mask(frame, mask):
    """ Apply the selected mask to the image. """
    return frame * mask[:, :, np.newaxis]

def process_frame(frame):
    """ Converts frame to tensor, applies model, and returns inpainted output. """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Convert OpenCV (NumPy) image to PIL Image
    image_pil = Image.fromarray(image)

    # Generate the mask
    mask = generate_blob_mask(np.array(image_pil))

    # Convert mask to the correct shape
    mask = cv2.resize(mask, (256, 256))  # Ensure mask matches the image size

    # Convert to tensor
    image_tensor = transform(image_pil).unsqueeze(0).to(device)  # Now it's PIL format!
    mask_tensor = torch.tensor(mask / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Perform inpainting
    with torch.no_grad():
        inpainted_tensor = model(image_tensor * mask_tensor, mask_tensor)

    # Convert back to NumPy
    inpainted_image = inpainted_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    inpainted_image = np.clip(inpainted_image, 0, 1)

    return image, mask, inpainted_image

def show_result(image, mask, inpainted):
    """ Display results in a new matplotlib figure. """
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

    # Resize and apply mask for visualization
    display_frame = cv2.resize(frame, (256, 256))
    
    # Generate the mask using blob generator
    mask = generate_blob_mask(display_frame)
    
    # Apply the mask to the frame
    masked_frame = apply_mask(display_frame, mask)

    # Show webcam feed with mask overlay
    cv2.imshow("Live Inpainting (Press 'C' to capture, 'Q' to quit)", masked_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Capture and process frame
        print("Capturing and processing frame...")
        image, mask, inpainted = process_frame(frame)
        show_result(image, mask, inpainted)

    elif key == ord('q'):  # Quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()