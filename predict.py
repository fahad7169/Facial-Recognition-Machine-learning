import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('face_recognition_model.keras')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the pre-trained face detector (Haar Cascade for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the image before prediction
def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to match the input size of the model (224x224)
    gray_resized = cv2.resize(gray, (224, 224))
    
    # Normalize the image
    gray_resized = gray_resized.astype('float32') / 255.0
    
    # Expand dimensions to match the input shape of the model (batch size, height, width, channels)
    gray_resized = np.expand_dims(gray_resized, axis=-1)  # Add channel dimension for grayscale
    
    return np.expand_dims(gray_resized, axis=0)  # Add batch size dimension

# Function to predict the label of the given image
def predict_image(face_image):
    # Preprocess the face image
    img = preprocess_image(face_image)
    
    # Make a prediction using the model
    prediction = model.predict(img)
    
    # Get the predicted class index (the class with the highest probability)
    predicted_class_index = np.argmax(prediction)
    
    # Convert the predicted class index back to the original label (e.g., student's name)
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
    
    # Get the confidence score (probability of the predicted class)
    confidence = prediction[0][predicted_class_index]
    
    return predicted_label, confidence

# Only run the real-time capture if this file is executed directly
if __name__ == "__main__":
    # Start real-time webcam capture
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop the face from the frame
            face_image = frame[y:y + h, x:x + w]

            # Predict the label (name) for the detected face
            predicted_label, confidence = predict_image(face_image)

            # Define a confidence threshold to consider a prediction valid
            confidence_threshold = 0.65

            # If the confidence is above the threshold, show the predicted name; otherwise, show "Unknown"
            if confidence >= confidence_threshold:
                label = f"{predicted_label} ({confidence * 100:.2f}%)"
            else:
                label = "Unknown"

            # Display the label below the rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label, (x, y - 10), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the webcam feed with the bounding box and label
        cv2.imshow('Real-Time Face Recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
