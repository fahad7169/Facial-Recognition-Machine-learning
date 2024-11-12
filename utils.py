import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

def preprocess_face(face_img, img_size=(224, 224)):
    face_img = cv2.resize(face_img, img_size)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=(0, -1))
    return face_img

def find_closest_match(face_img, model, reference_embeddings, threshold=0.5):
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
