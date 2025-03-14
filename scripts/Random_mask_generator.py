import cv2
import numpy as np
import random
import os
from glob import glob

def generate_skeleton_mask(height, width, max_mask_fraction=0.4):
    """
    Generates a mask with a random skeleton that grows by dilation.

    :param height: Height of the image.
    :param width: Width of the image.
    :param max_mask_fraction: Maximum fraction of the image to be masked.
    :return: A binary mask.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Start with a sparse set of random points as the skeleton
    num_seeds = random.randint(5, 15)
    for _ in range(num_seeds):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        mask[y, x] = 255

    total_pixels = height * width
    masked_pixels = np.sum(mask == 255)

    # Grow the skeleton using morphological dilation
    while masked_pixels / total_pixels < max_mask_fraction:
        # Random kernel size for dilation to create organic growth
        kernel_size = random.randint(3, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

        masked_pixels = np.sum(mask == 255)

    return mask

def apply_mask(image, mask, fill_color=None):
    """
    Applies a generated mask to an image.

    :param image: Input image as a NumPy array.
    :param mask: Mask to be applied.
    :param fill_color: Color to fill the masked regions (None means random choice between white or black).
    :return: Masked image.
    """
    if fill_color is None:
        fill_color = random.choice([0, 255])  # Randomly choose white or black fill

    masked_image = image.copy()
    masked_image[mask == 255] = fill_color

    return masked_image

def process_images(input_folder, output_folder, max_mask_fraction=0.4, fill_color=None):
    """
    Processes all images in a folder, applies skeleton-based masks, and saves them.

    :param input_folder: Path to the folder containing images.
    :param output_folder: Path to the folder where masked images will be saved.
    :param max_mask_fraction: Maximum fraction of the image that can be masked.
    :param fill_color: Color to fill the masked regions (None means random choice between white or black).
    """
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob(os.path.join(input_folder, "*.*"))  # Adjust extensions if needed

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping {image_path}, could not load image.")
            continue

        height, width, _ = image.shape
        mask = generate_skeleton_mask(height, width, max_mask_fraction)
        masked_image = apply_mask(image, mask, fill_color)

        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, masked_image)
        print(f"Saved masked image: {output_path}")

if __name__ == "__main__":
    input_folder = "input_images"  # Folder containing original images
    output_folder = "masked_images"  # Folder to save masked images

    process_images(input_folder, output_folder, max_mask_fraction=0.4, fill_color=None)
