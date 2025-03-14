import os
import cv2
import skimage.exposure
import numpy as np
from numpy.random import default_rng

def generate_blob_mask(image_path, save_path, sigma=15, threshold=175, seedval=75):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    rng = default_rng(seed=seedval)

    noise = rng.integers(0, 255, (height, width), np.uint8, True)
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)
    thresh = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask_filename = os.path.join(save_path, os.path.basename(image_path).replace('.jpg', '_mask.png'))
    cv2.imwrite(mask_filename, mask)

image_folder = "dataset/images"
mask_folder = "dataset/masks"
os.makedirs(mask_folder, exist_ok=True)

for image_file in os.listdir(image_folder):
    if image_file.endswith(".jpg"):
        generate_blob_mask(os.path.join(image_folder, image_file), mask_folder)

print("Masks saved successfully!")
