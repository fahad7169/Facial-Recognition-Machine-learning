#utils.py
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import dlib
import os

class FacePreprocessor:
    def __init__(self, predictor_path=None):
        """
        Initialize face preprocessor with optional facial landmark detector.
        
        Parameters:
            predictor_path (str): Path to dlib's shape predictor file
        """
        self.predictor = None
        self.detector = None
        
        # Try to initialize dlib face detector and landmark predictor
        try:
            self.detector = dlib.get_frontal_face_detector()
            if predictor_path and os.path.exists(predictor_path):
                self.predictor = dlib.shape_predictor(predictor_path)
                print("Loaded dlib facial landmark predictor")
            else:
                print("dlib predictor not available, using basic preprocessing")
        except ImportError:
            print("dlib not available, using basic preprocessing")
    
    def detect_face_landmarks(self, image):
        """
        Detect facial landmarks using dlib.
        
        Parameters:
            image (numpy array): Input image
            
        Returns:
            list: List of facial landmark points
        """
        if self.detector is None or self.predictor is None:
            return None
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            
            if len(faces) > 0:
                # Get landmarks for the first face
                landmarks = self.predictor(gray, faces[0])
                points = []
                
                # Extract 68 landmark points
                for i in range(68):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    points.append((x, y))
                
                return points
            
        except Exception as e:
            print(f"Error detecting landmarks: {e}")
        
        return None
    
    def align_face(self, image, landmarks=None, target_size=(224, 224)):
        """
        Align face using facial landmarks for better recognition.
        
        Parameters:
            image (numpy array): Input image
            landmarks (list): Facial landmark points
            target_size (tuple): Target size for output image
            
        Returns:
            numpy array: Aligned face image
        """
        if landmarks is None:
            # If no landmarks, just resize the image
            return cv2.resize(image, target_size)
        
        try:
            # Get eye centers (landmarks 36-45 for left eye, 42-47 for right eye)
            left_eye_center = np.mean(landmarks[36:42], axis=0)
            right_eye_center = np.mean(landmarks[42:48], axis=0)
            
            # Calculate angle for rotation
            eye_angle = np.degrees(np.arctan2(
                right_eye_center[1] - left_eye_center[1],
                right_eye_center[0] - left_eye_center[0]
            ))
            
            # Get center of image
            center = (image.shape[1] // 2, image.shape[0] // 2)
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, eye_angle, 1.0)
            
            # Apply rotation
            rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            
            # Crop and resize
            aligned = cv2.resize(rotated, target_size)
            
            return aligned
            
        except Exception as e:
            print(f"Error aligning face: {e}")
            return cv2.resize(image, target_size)
    
    def enhance_face(self, image):
        """
        Apply basic image enhancement for better face recognition.
        
        Parameters:
            image (numpy array): Input image
            
        Returns:
            numpy array: Enhanced image
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return image
    
    def preprocess_for_embedding(self, image, target_size=(224, 224), 
                                use_landmarks=True, enhance=True):
        """
        Complete preprocessing pipeline for face embedding.
        
        Parameters:
            image (numpy array): Input image
            target_size (tuple): Target size for output
            use_landmarks (bool): Whether to use facial landmarks for alignment
            enhance (bool): Whether to apply image enhancement
            
        Returns:
            numpy array: Preprocessed image ready for embedding
        """
        try:
            # Detect landmarks if requested
            landmarks = None
            if use_landmarks and self.predictor is not None:
                landmarks = self.detect_face_landmarks(image)
            
            # Align face
            aligned = self.align_face(image, landmarks, target_size)
            
            # Apply enhancement if requested
            if enhance:
                aligned = self.enhance_face(aligned)
            
            # Normalize to [0, 1]
            normalized = aligned.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Fallback to basic preprocessing
            resized = cv2.resize(image, target_size)
            normalized = resized.astype(np.float32) / 255.0
            return normalized

def create_face_dataset(image_directory, output_directory, preprocessor=None):
    """
    Create a preprocessed face dataset from raw images.
    
    Parameters:
        image_directory (str): Directory containing raw images
        output_directory (str): Directory to save preprocessed images
        preprocessor (FacePreprocessor): Face preprocessor instance
    """
    if preprocessor is None:
        preprocessor = FacePreprocessor()
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    processed_count = 0
    
    for person_name in os.listdir(image_directory):
        person_path = os.path.join(image_directory, person_name)
        if os.path.isdir(person_path):
            # Create output directory for this person
            output_person_path = os.path.join(output_directory, person_name)
            if not os.path.exists(output_person_path):
                os.makedirs(output_person_path)
            
            for filename in os.listdir(person_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    input_path = os.path.join(person_path, filename)
                    output_path = os.path.join(output_person_path, filename)
                    
                    try:
                        # Load image
                        image = cv2.imread(input_path)
                        if image is not None:
                            # Preprocess image
                            processed = preprocessor.preprocess_for_embedding(
                                image, 
                                target_size=(224, 224),
                                use_landmarks=True,
                                enhance=True
                            )
                            
                            # Convert back to uint8 for saving
                            processed_uint8 = (processed * 255).astype(np.uint8)
                            
                            # Save processed image
                            cv2.imwrite(output_path, processed_uint8)
                            processed_count += 1
                            
                    except Exception as e:
                        print(f"Error processing {input_path}: {e}")
    
    print(f"Processed {processed_count} images")

def calculate_embedding_similarity(embedding1, embedding2, method='cosine'):
    """
    Calculate similarity between two face embeddings.
    
    Parameters:
        embedding1, embedding2 (numpy arrays): Face embeddings
        method (str): Similarity method ('cosine' or 'euclidean')
        
    Returns:
        float: Similarity score
    """
    if method == 'cosine':
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    elif method == 'euclidean':
        distance = np.linalg.norm(embedding1 - embedding2)
        return 1.0 / (1.0 + distance)  # Convert to similarity score
    else:
        raise ValueError("Method must be 'cosine' or 'euclidean'")

def find_best_matches(query_embedding, known_embeddings, top_k=5, threshold=0.5):
    """
    Find the best matches for a query embedding.
    
    Parameters:
        query_embedding (numpy array): Query face embedding
        known_embeddings (dict): Dictionary of known face embeddings
        top_k (int): Number of top matches to return
        threshold (float): Minimum similarity threshold
        
    Returns:
        list: List of (person_name, similarity_score) tuples
    """
    similarities = []
    
    for person_name, embedding in known_embeddings.items():
        similarity = calculate_embedding_similarity(query_embedding, embedding, method='cosine')
        if similarity >= threshold:
            similarities.append((person_name, similarity))
    
    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

def validate_face_recognition_system(embedding_model, test_images, known_embeddings, threshold=0.6):
    """
    Validate the face recognition system performance.
    
    Parameters:
        embedding_model: Trained embedding model
        test_images (dict): Dictionary of test images by person
        known_embeddings (dict): Known face embeddings
        threshold (float): Recognition threshold
        
    Returns:
        dict: Validation results
    """
    results = {
        'total_tests': 0,
        'correct_recognitions': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'accuracy': 0.0
    }
    
    for person_name, images in test_images.items():
        if person_name in known_embeddings:
            known_embedding = known_embeddings[person_name]
            
            for image in images:
                results['total_tests'] += 1
                
                # Generate embedding for test image
                test_embedding = embedding_model.predict(np.expand_dims(image, axis=0), verbose=0).flatten()
                
                # Calculate similarity
                similarity = calculate_embedding_similarity(test_embedding, known_embedding)
                
                # Check if recognition is correct
                if similarity >= threshold:
                    results['correct_recognitions'] += 1
                else:
                    results['false_negatives'] += 1
    
    # Calculate accuracy
    if results['total_tests'] > 0:
        results['accuracy'] = results['correct_recognitions'] / results['total_tests']
    
    return results
