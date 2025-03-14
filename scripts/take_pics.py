import cv2
import os
import time

# Ensure the dataset folder exists
output_dir = "dataset/images"
os.makedirs(output_dir, exist_ok=True)

# Initialize webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

image_counter = 1
max_images = 10_000  # Capture 10,000 images
capture_interval = 0.0005  # Time in seconds between captures

print("Starting automatic image capture. Press 'q' to stop early.")

while image_counter <= max_images:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Resize the image to 256x256
    resized_image = cv2.resize(frame, (256, 256))

    # Save the image with a unique filename
    img_filename = f"{output_dir}/image{image_counter}.jpg"
    cv2.imwrite(img_filename, resized_image)
    print(f"Image {image_counter} saved to {img_filename}")

    image_counter += 1

    # Display the captured frame in a window
    cv2.imshow('Webcam Feed', frame)

    # Wait for 'q' to be pressed to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Early exit triggered. Exiting...")
        break

    # Wait before capturing the next image
    time.sleep(capture_interval)

print("Image capture complete. Exiting...")

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
