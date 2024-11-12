import cv2
import os
import numpy as np

# Initialize the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def resize_with_aspect_ratio(img, target_size=(224, 224), color=(0, 0, 0)):
    target_width, target_height = target_size
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Determine new dimensions to maintain aspect ratio
    if aspect_ratio > 1:  # Image is wider than tall
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:  # Image is taller than wide
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_img = cv2.resize(img, (new_width, new_height))
    padded_img = np.full((target_height, target_width, 3), color, dtype=np.uint8)

    # Center the resized image within the padded image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

    return padded_img

def process_and_save_faces(input_folder, output_folder, target_size=(224, 224)):
    for student_name in os.listdir(input_folder):
        student_folder = os.path.join(input_folder, student_name)
        output_student_folder = os.path.join(output_folder, student_name)

        # Create output folder if it doesn't exist
        if not os.path.exists(output_student_folder):
            os.makedirs(output_student_folder)

        for img_name in os.listdir(student_folder):
            img_path = os.path.join(student_folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            # Convert to grayscale for face detection
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            
            # Process each face found
            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y + h, x:x + w]  # Crop the face region

                # Resize with padding to keep aspect ratio
                face_resized_padded = resize_with_aspect_ratio(face, target_size)

                # Convert the resized, padded face image to grayscale
                face_gray = cv2.cvtColor(face_resized_padded, cv2.COLOR_BGR2GRAY)

                # Save the processed face image
                output_path = os.path.join(output_student_folder, f"{img_name}_face_{i}.jpg")
                cv2.imwrite(output_path, face_gray)
                print(f"Processed and saved face image: {output_path}")

# Define input and output folders
input_folder = 'dataset'
output_folder = 'resized_images'

# Run the function
process_and_save_faces(input_folder, output_folder)
