import os
import cv2
import skimage.exposure
import numpy as np
from numpy.random import default_rng

def generate_blob_mask(image_path, save_path, sigma=15, threshold=175, min_blob_area_pct=0.05, max_blob_area_pct=0.10):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return

    height, width = img.shape[:2]
    
    # Calculate the area of the image
    total_area = height * width

    # Set the blob size area between 5% to 10% of the image area
    min_area = int(min_blob_area_pct * total_area)
    max_area = int(max_blob_area_pct * total_area)

    # Generate a unique seed based on the filename (or use the current time)
    seedval = hash(os.path.basename(image_path)) % (2**32)  # Unique seed for each image
    rng = default_rng(seed=seedval)

    # Generate noise for the mask
    noise = rng.integers(0, 255, (height, width), np.uint8, True)
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)

    # Threshold to create the binary mask
    thresh = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Randomize the number of blobs
    num_blobs = rng.integers(1, 5)  # Random number of blobs between 1 and 5

    for _ in range(num_blobs):
        # Randomly determine the size of the blob (area between min_area and max_area)
        blob_area = rng.integers(min_area, max_area)

        # Randomly determine the height of the blob (between 5 and height//4)
        blob_height = rng.integers(5, height // 4)  

        # Calculate the corresponding width of the blob
        blob_width = blob_area // blob_height

        # Ensure the width and height do not exceed image dimensions
        blob_height = min(blob_height, height - 1)  # Avoid exceeding image height
        blob_width = min(blob_width, width - 1)    # Avoid exceeding image width

        # Ensure there's enough space for the blob
        if width - blob_width <= 0 or height - blob_height <= 0:
            continue  # Skip if no space for the blob

        # Randomly place the blob, ensuring it fits within the image bounds
        y = rng.integers(0, height - blob_height)
        x = rng.integers(0, width - blob_width)

        # Draw the rectangle blob on the mask
        mask[y:y+blob_height, x:x+blob_width] = 255

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)

    mask_filename = os.path.join(save_path, os.path.basename(image_path).replace('.jpg', '_mask.png'))
    cv2.imwrite(mask_filename, mask)

# Define dataset paths
image_root = "dataset/images"
mask_root = "dataset/masks"

# Iterate through training and validation sets
for split in ["training", "validation"]:
    for resized_split in ["", "_resized"]:
        image_folder = os.path.join(image_root, f"{split}{resized_split}")
        mask_folder = os.path.join(mask_root, f"{split}{resized_split}")
        os.makedirs(mask_folder, exist_ok=True)

        for image_file in os.listdir(image_folder):
            if image_file.endswith(".jpg"):  # Adjust for other formats if needed
                generate_blob_mask(os.path.join(image_folder, image_file), mask_folder)

print("Masks saved successfully for training and validation sets!")
