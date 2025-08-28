#predict.py
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from model import cosine_similarity, euclidean_distance
import os

class FaceRecognitionSystem:
    def __init__(self, model_path='face_embedding_model.keras', 
                 embeddings_path='known_face_embeddings.pkl',
                 person_names_path='person_names.pkl'):
        """
        Initialize the face recognition system.
        
        Parameters:
            model_path (str): Path to the trained embedding model
            embeddings_path (str): Path to the known face embeddings
            person_names_path (str): Path to the person names list
        """
        self.model = None
        self.known_embeddings = {}
        self.person_names = []
        self.face_cascade = None
        self.similarity_threshold = 0.6
        
        # Load the model and data
        self._load_model_and_data(model_path, embeddings_path, person_names_path)
        
        # Initialize face detector
        self._initialize_face_detector()
    
    def _load_model_and_data(self, model_path, embeddings_path, person_names_path):
        """Load the trained model and known face data."""
        try:
            # Load the embedding model
            if os.path.exists(model_path):
                self.model = load_model(model_path, compile=False)
                print(f"Loaded embedding model from {model_path}")
            else:
                print(f"Warning: Model file {model_path} not found")
                return
            
            # Load known face embeddings
            if os.path.exists(embeddings_path):
                with open(embeddings_path, 'rb') as f:
                    self.known_embeddings = pickle.load(f)
                print(f"Loaded {len(self.known_embeddings)} known face embeddings")
            else:
                print(f"Warning: Embeddings file {embeddings_path} not found")
            
            # Load person names
            if os.path.exists(person_names_path):
                with open(person_names_path, 'rb') as f:
                    self.person_names = pickle.load(f)
                print(f"Loaded {len(self.person_names)} person names")
            else:
                print(f"Warning: Person names file {person_names_path} not found")
                
        except Exception as e:
            print(f"Error loading model and data: {e}")
    
    def _initialize_face_detector(self):
        """Initialize the face detection cascade classifier."""
        try:
            # Try to load a more accurate face detector
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(face_cascade_path):
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
                print("Loaded Haar cascade face detector")
            else:
                print("Warning: Haar cascade file not found")
        except Exception as e:
            print(f"Error initializing face detector: {e}")
    
    def preprocess_face(self, face_img, target_size=(224, 224)):
        """
        Preprocess a face image for embedding generation.
        
        Parameters:
            face_img (numpy array): Input face image
            target_size (tuple): Target size for the image
        
        Returns:
            numpy array: Preprocessed image
        """
        try:
            # Convert BGR to RGB
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            face_img = cv2.resize(face_img, target_size)
            
            # Normalize to [0, 1]
            face_img = face_img.astype(np.float32) / 255.0
            
            # Add batch dimension
            face_img = np.expand_dims(face_img, axis=0)
            
            return face_img
            
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def generate_embedding(self, face_img):
        """
        Generate embedding for a face image.
        
        Parameters:
            face_img (numpy array): Preprocessed face image
        
        Returns:
            numpy array: Face embedding vector
        """
        try:
            if self.model is None:
                print("Model not loaded")
                return None
            
            # Generate embedding
            embedding = self.model.predict(face_img, verbose=0)
            return embedding.flatten()
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def recognize_face(self, face_img):
        """
        Recognize a face by comparing its embedding to known embeddings.
        
        Parameters:
            face_img (numpy array): Face image to recognize
        
        Returns:
            tuple: (person_name, confidence_score, similarity_method)
        """
        try:
            # Preprocess the face
            processed_face = self.preprocess_face(face_img)
            if processed_face is None:
                return "Unknown", 0.0, "error"
            
            # Generate embedding
            face_embedding = self.generate_embedding(processed_face)
            if face_embedding is None:
                return "Unknown", 0.0, "error"
            
            if not self.known_embeddings:
                return "Unknown", 0.0, "no_known_faces"
            
            # Compare with known embeddings
            best_match = None
            best_score = -1
            best_method = ""
            
            for person_name, known_embedding in self.known_embeddings.items():
                # Calculate cosine similarity
                cosine_sim = cosine_similarity(face_embedding, known_embedding)
                
                # Calculate Euclidean distance (lower is better)
                euclidean_dist = euclidean_distance(face_embedding, known_embedding)
                euclidean_sim = 1.0 / (1.0 + euclidean_dist)  # Convert to similarity score
                
                # Use the better of the two similarity measures
                if cosine_sim > euclidean_sim:
                    score = cosine_sim
                    method = "cosine"
                else:
                    score = euclidean_sim
                    method = "euclidean"
                
                if score > best_score:
                    best_score = score
                    best_match = person_name
                    best_method = method
            
            # Apply threshold
            if best_score >= self.similarity_threshold:
                return best_match, best_score, best_method
            else:
                return "Unknown", best_score, best_method
                
        except Exception as e:
            print(f"Error recognizing face: {e}")
            return "Unknown", 0.0, "error"
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame.
        
        Parameters:
            frame (numpy array): Input frame
        
        Returns:
            list: List of face bounding boxes (x, y, w, h)
        """
        try:
            if self.face_cascade is None:
                return []
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            return faces
            
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def process_frame(self, frame):
        """
        Process a frame for face recognition.
        
        Parameters:
            frame (numpy array): Input frame
        
        Returns:
            numpy array: Processed frame with recognition results
        """
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract face region
                face_img = frame[y:y + h, x:x + w]
                
                # Recognize face
                person_name, confidence, method = self.recognize_face(face_img)
                
                # Display results
                if person_name != "Unknown":
                    label = f"{person_name} ({confidence:.2f})"
                    color = (0, 255, 0)  # Green for recognized
                else:
                    label = f"Unknown ({confidence:.2f})"
                    color = (0, 0, 255)  # Red for unknown
                
                # Add method info
                method_label = f"Method: {method}"
                
                # Display labels
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, method_label, (x, y + h + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame
    
    def run_realtime_recognition(self):
        """Run real-time face recognition from webcam."""
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open webcam")
                return
            
            print("Press 'q' to quit, 's' to save current frame")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Face Recognition System', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"captured_frame_{len(os.listdir('.'))}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Saved frame as {filename}")
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error in real-time recognition: {e}")

def predict_image(face_image):
    """
    Legacy function for backward compatibility.
    
    Parameters:
        face_image (numpy array): Face image to recognize
    
    Returns:
        tuple: (person_name, confidence_score)
    """
    # Create a temporary recognition system
    temp_system = FaceRecognitionSystem()
    
    # Recognize the face
    person_name, confidence, method = temp_system.recognize_face(face_image)
    
    return person_name, confidence

if __name__ == "__main__":
    # Create and run the face recognition system
    face_system = FaceRecognitionSystem()
    face_system.run_realtime_recognition()
