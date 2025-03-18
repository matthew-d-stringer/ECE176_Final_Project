import cv2
import os
import time

output_dir = "dataset/images"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

image_counter = 1
max_images = 10000
capture_interval = 0.0005  

print("Starting automatic image capture. Press 'q' to stop early.")

while image_counter <= max_images:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Resize the image to 256x256
    resized_image = cv2.resize(frame, (256, 256))

    print(f"Image {image_counter} saved to {img_filename}")

    image_counter += 1

    cv2.imshow('Webcam Feed', frame)

    # wait for q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Early exit triggered. Exiting...")
        break

    time.sleep(capture_interval)

print("Image capture complete. Exiting...")

cap.release()
cv2.destroyAllWindows()
