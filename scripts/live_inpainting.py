import torch
<<<<<<< HEAD
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
# Add the src/ directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add src/ directory to Python path
sys.path.append(os.path.join(project_root, "src"))

# Now import the modules
from inpainting_model import InpaintingModel
# from partial_convolution2d import PartialConvolution2d
=======
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inpainting_model import InpaintingModel  # Ensure this is your trained model class
from PIL import Image
>>>>>>> 64d76b776681c1df2b20aa1cdab80f8ec68294b0

# Load trained model
MODEL_PATH = "checkpoints/inpainting_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InpaintingModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Blob Mask Generator Code
def generate_blob_mask(image, sigma=15, threshold=175):
    height, width = image.shape[:2]

    # Generate random noise
    noise = np.random.randint(0, 255, (height, width), dtype=np.uint8)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
    
    # Stretch intensity
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)
    
    # Threshold to create binary mask
    thresh = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY)[1]

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

<<<<<<< HEAD
# Mask options (circle, rectangle, and blob mask)
mask_options = [
    lambda h, w: cv2.circle(np.ones((h, w)), (w//2, h//2), w//4, 0, -1),  # Centered circle mask
    lambda h, w: cv2.rectangle(np.ones((h, w)), (w//4, h//4), (3*w//4, 3*h//4), 0, -1),  # Square mask
    generate_blob_mask  # Small noise (blob) mask
]
mask_index = 0  # Default mask selection (circle)

=======
>>>>>>> 64d76b776681c1df2b20aa1cdab80f8ec68294b0
# Initialize webcam
cap = cv2.VideoCapture(0)

def apply_mask(frame, mask):
    """ Apply the selected mask to the image. """
    return frame * mask[:, :, np.newaxis]

def process_frame(frame):
    """ Converts frame to tensor, applies model, and returns inpainted output. """
<<<<<<< HEAD
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    
    # Generate the mask using the selected mask generator
    mask = mask_options[mask_index](256, 256)
    
    # Convert to tensor
    image_tensor = transform(image).unsqueeze(0).to(device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
=======
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
>>>>>>> 64d76b776681c1df2b20aa1cdab80f8ec68294b0

    # Perform inpainting
    with torch.no_grad():
        inpainted_tensor = model(image_tensor * mask_tensor, mask_tensor)

<<<<<<< HEAD
    # Convert back to numpy
=======
    # Convert back to NumPy
>>>>>>> 64d76b776681c1df2b20aa1cdab80f8ec68294b0
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
    
<<<<<<< HEAD
    # Generate the mask using the selected mask generator
    mask = mask_options[mask_index](256, 256)
=======
    # Generate the mask using blob generator
    mask = generate_blob_mask(display_frame)
>>>>>>> 64d76b776681c1df2b20aa1cdab80f8ec68294b0
    
    # Apply the mask to the frame
    masked_frame = apply_mask(display_frame, mask)

    # Show webcam feed with mask overlay
<<<<<<< HEAD
    cv2.imshow("Live Inpainting (Press 'M' to change mask, 'C' to capture, 'Q' to quit)", masked_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('m'):  # Switch mask
        mask_index = (mask_index + 1) % len(mask_options)
        print(f"Switched to mask {mask_index+1}")

    elif key == ord('c'):  # Capture and process frame
=======
    cv2.imshow("Live Inpainting (Press 'C' to capture, 'Q' to quit)", masked_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Capture and process frame
>>>>>>> 64d76b776681c1df2b20aa1cdab80f8ec68294b0
        print("Capturing and processing frame...")
        image, mask, inpainted = process_frame(frame)
        show_result(image, mask, inpainted)

    elif key == ord('q'):  # Quit
        break

# Cleanup
cap.release()
<<<<<<< HEAD
cv2.destroyAllWindows()
=======
cv2.destroyAllWindows()
>>>>>>> 64d76b776681c1df2b20aa1cdab80f8ec68294b0
