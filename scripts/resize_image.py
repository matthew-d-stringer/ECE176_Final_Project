import os
import cv2

def resize_and_save_images(input_dir, output_dir, size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img_resized)
    
    print(f"Resized images saved to {output_dir}")

for split in ["training", "validation"]:
    resize_and_save_images(f"dataset/images/{split}", f"dataset/images/{split}_resized")
    resize_and_save_images(f"dataset/masks/{split}", f"dataset/masks/{split}_resized", size=(256, 256))  # Resize masks too