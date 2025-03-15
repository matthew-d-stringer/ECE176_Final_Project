import os
import cv2
import skimage.exposure
import numpy as np
from numpy.random import default_rng

def generate_blob_mask(image_path, save_path, sigma=15, threshold=175):
    img = cv2.imread(image_path)
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

    mask_filename = os.path.join(save_path, os.path.basename(image_path).replace('.jpg', '_mask.png'))
    cv2.imwrite(mask_filename, mask)

image_folder = "dataset/images/training"
mask_folder = "dataset/images/training_masks"
os.makedirs(mask_folder, exist_ok=True)

for image_file in os.listdir(image_folder):
    if image_file.endswith(".jpg"):
        generate_blob_mask(os.path.join(image_folder, image_file), mask_folder)

print("Black blobs on white background saved successfully!")
