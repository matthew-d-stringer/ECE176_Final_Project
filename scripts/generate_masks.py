import os
import cv2
import skimage.exposure
import numpy as np
from numpy.random import default_rng

def generate_blob_mask(image_path, save_path, sigma=15, threshold=175):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return

    height, width = img.shape[:2]

    # Generate a unique seed based on the filename (or use the current time)
    seedval = hash(os.path.basename(image_path)) % (2**32)  # Unique seed for each image
    rng = default_rng(seed=seedval)

    noise = rng.integers(0, 255, (height, width), np.uint8, True)
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)
    
    # **Use THRESH_BINARY_INV to make blobs black and background white**
    thresh = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)

    mask_filename = os.path.join(save_path, os.path.basename(image_path).replace('.jpg', '_mask.png'))
    cv2.imwrite(mask_filename, mask)

# Define dataset paths
image_root = "dataset/images"
mask_root = "dataset/masks"

# Iterate through both original and resized datasets (training and validation)
for split in ["training", "validation"]:
    for resized in ["_resized"]:
        # Check if the image folder exists
        image_folder = os.path.join(image_root, split + resized)
        if not os.path.exists(image_folder):
            print(f"Warning: {image_folder} does not exist. Skipping this folder.")
            continue
        
        # Set the corresponding mask folder
        mask_folder = os.path.join(mask_root, split + resized)
        os.makedirs(mask_folder, exist_ok=True)

        # Iterate over the images in the folder
        for image_file in os.listdir(image_folder):
            if image_file.endswith(".jpg"):  # Adjust for other formats if needed
                generate_blob_mask(os.path.join(image_folder, image_file), mask_folder)

print("Masks saved successfully for training and validation sets (including resized ones)!")