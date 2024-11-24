#utils.py
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

def preprocess_face(face_img, img_size=(224, 224)):
    """
    Preprocess a face image for input to the face recognition model.

    The function takes a face image in BGR format, resizes it to the input size of the model (224x224 by default),
    converts it to grayscale, normalizes the image to the range [0, 1], and adds singleton dimensions to the beginning
    and end of the image to match the expected input shape of the model.

    Args:
        face_img (numpy array): Input face image in BGR format.
        img_size (tuple of int, optional): Input size of the model. Defaults to (224, 224).

    Returns:
        numpy array: Preprocessed face image with shape (1, *img_size, 1).
    """
    face_img = cv2.resize(face_img, img_size)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=(0, -1))
    return face_img

def find_closest_match(face_img, model, reference_embeddings, threshold=0.5):
    """
    Find the closest matching label for a given face image.

    This function computes the embedding for a given face image using a pre-trained model
    and compares it against reference embeddings to find the closest match. If the minimum
    distance is below a specified threshold, the corresponding label is returned; otherwise,
    the label "Unknown" is returned. The confidence score is calculated as one minus the 
    minimum distance.

    Args:
        face_img (numpy array): The input face image, preprocessed to model input shape.
        model (keras.Model): The pre-trained model used to compute face embeddings.
        reference_embeddings (dict): Dictionary with labels as keys and reference embeddings as values.
        threshold (float): Distance threshold to determine if a match is close enough; defaults to 0.5.

    Returns:
        tuple: A tuple containing the best matching label (str) and the confidence score (float).
    """
    embedding = model.predict(face_img)[0]
    min_distance = float('inf')
    best_label = "Unknown"
    
    for label, ref_embedding in reference_embeddings.items():
        distance = np.linalg.norm(embedding - ref_embedding)
        
        if distance < min_distance:
            min_distance = distance
            best_label = label if distance < threshold else "Unknown"
    
    confidence = 1 - min_distance
    return best_label, confidence
