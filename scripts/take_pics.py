import cv2
import os

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

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display the captured frame in a window
    cv2.imshow('Webcam Feed', frame)

    # Wait for key press and capture image if spacebar is pressed
    key = cv2.waitKey(1) & 0xFF  # Capture key press
    
    if key == 32:  # Spacebar pressed
        # Resize the image to 256x256
        resized_image = cv2.resize(frame, (256, 256))

        # Save the image with a unique filename
        img_filename = f"{output_dir}/image_{image_counter}.jpg"
        cv2.imwrite(img_filename, resized_image)

        print(f"Image {image_counter} saved to {img_filename}")
        image_counter += 1

    elif key == ord('q'):  # Press 'q' to exit
        print("Exiting...")
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
