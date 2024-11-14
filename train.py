#trai.py
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Function to load images and labels
def load_images_from_directory(directory):
    images = []
    labels = []
    label_encoder = LabelEncoder()

    # Loop through each directory (which represents a class)
    for label in os.listdir(directory):
        class_path = os.path.join(directory, label)
        
        if os.path.isdir(class_path):
            # Loop through each image in the class folder
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                if filename.endswith(".jpg") or filename.endswith(".png"):  # You can add more extensions if needed
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
                    img = cv2.resize(img, (224, 224))  # Resize to 224x224 (or any size you need)
                    images.append(img)
                    labels.append(label)
    
    # Convert to numpy arrays
    images = np.array(images, dtype='float32')
    labels = np.array(labels)

    # Normalize images (optional but recommended)
    images /= 255.0  # Normalize pixel values to the range [0, 1]
    
    # Convert labels to numeric format using LabelEncoder
    labels = label_encoder.fit_transform(labels)
    
    return images, labels, label_encoder

# Function to train the model
def train_cnn_model(directory):
    # Load images and labels
    images, labels, label_encoder = load_images_from_directory(directory)

    # Build the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))  # Grayscale images
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(labels)), activation='softmax'))  # Output layer for classification

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


   # Train the model with early stopping
    model.fit(images, labels, epochs=15, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

    # Save the trained model
    model.save('face_recognition_model.keras')
    print("Model saved to 'face_recognition_model.keras'")

    # Save the label encoder to a file using pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved to 'label_encoder.pkl'")

# Call the training function with the directory where images are stored
train_cnn_model('resized_images')
