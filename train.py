#train.py
import os
import numpy as np
import cv2
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from model import create_embedding_model
import random

def load_and_preprocess_images(directory, target_size=(224, 224)):
    """
    Load and preprocess images for training the embedding model.
    
    Parameters:
        directory (str): Path to directory containing person folders
        target_size (tuple): Target size for images (height, width)
    
    Returns:
        dict: Dictionary mapping person names to lists of preprocessed images
    """
    person_images = {}
    
    for person_name in os.listdir(directory):
        person_path = os.path.join(directory, person_name)
        if os.path.isdir(person_path):
            person_images[person_name] = []
            
            for filename in os.listdir(person_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_path, filename)
                    img = cv2.imread(img_path)
                    
                    if img is not None:
                        # Convert BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Resize image
                        img = cv2.resize(img, target_size)
                        
                        # Normalize to [0, 1]
                        img = img.astype(np.float32) / 255.0
                        
                        person_images[person_name].append(img)
    
    return person_images

def create_triplet_data(person_images, num_triplets=1000):
    """
    Create training triplets (anchor, positive, negative).
    
    Parameters:
        person_images (dict): Dictionary mapping person names to image lists
        num_triplets (int): Number of triplets to generate
    
    Returns:
        tuple: Arrays of anchor, positive, and negative images
    """
    anchors, positives, negatives = [], [], []
    
    person_names = list(person_images.keys())
    
    for _ in range(num_triplets):
        # Randomly select anchor person
        anchor_person = random.choice(person_names)
        
        # Get two different images of the same person (anchor and positive)
        if len(person_images[anchor_person]) >= 2:
            anchor_img, positive_img = random.sample(person_images[anchor_person], 2)
        else:
            # If only one image, duplicate it
            anchor_img = positive_img = person_images[anchor_person][0]
        
        # Get negative image from different person
        negative_person = random.choice([p for p in person_names if p != anchor_person])
        negative_img = random.choice(person_images[negative_person])
        
        anchors.append(anchor_img)
        positives.append(positive_img)
        negatives.append(negative_img)
    
    return np.array(anchors), np.array(positives), np.array(negatives)

def triplet_loss(alpha=0.2):
    """
    Triplet loss function for training face embeddings.
    
    Parameters:
        alpha (float): Margin parameter for the triplet loss
    
    Returns:
        function: Triplet loss function
    """
    def loss(y_true, y_pred):
        # y_true is not used in triplet loss
        # y_pred contains concatenated embeddings [anchor, positive, negative]
        
        # Split the concatenated embeddings
        embedding_size = y_pred.shape[-1] // 3
        anchor_emb = y_pred[:, :embedding_size]
        positive_emb = y_pred[:, embedding_size:2*embedding_size]
        negative_emb = y_pred[:, 2*embedding_size:]
        
        # Calculate distances
        pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=1)
        
        # Triplet loss: max(pos_dist - neg_dist + alpha, 0)
        basic_loss = pos_dist - neg_dist + alpha
        loss = tf.maximum(basic_loss, 0.0)
        
        return tf.reduce_mean(loss)
    
    return loss

def create_triplet_model(embedding_model, input_shape=(224, 224, 3)):
    """
    Create a triplet model for training face embeddings.
    
    Parameters:
        embedding_model: The embedding model
        input_shape (tuple): Input shape for images
    
    Returns:
        keras.Model: The triplet model
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Concatenate
    
    # Create three inputs for anchor, positive, and negative
    anchor_input = Input(shape=input_shape, name='anchor_input')
    positive_input = Input(shape=input_shape, name='positive_input')
    negative_input = Input(shape=input_shape, name='negative_input')
    
    # Generate embeddings for all three inputs
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)
    
    # Concatenate all embeddings for triplet loss
    concatenated_embeddings = Concatenate(axis=-1)([anchor_embedding, positive_embedding, negative_embedding])
    
    # Create the triplet model
    triplet_model = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=concatenated_embeddings
    )
    
    return triplet_model

def train_embedding_model(directory, epochs=50, batch_size=32):
    """
    Train the face embedding model using triplet loss.
    
    Parameters:
        directory (str): Path to directory containing person folders
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    """
    
    print("Loading and preprocessing images...")
    person_images = load_and_preprocess_images(directory)
    
    # Filter out people with too few images
    min_images_per_person = 5
    person_images = {k: v for k, v in person_images.items() if len(v) >= min_images_per_person}
    
    print(f"Found {len(person_images)} people with sufficient images")
    
    if len(person_images) < 2:
        print("Need at least 2 people with sufficient images for training")
        return None, None
    
    # Create training triplets
    print("Creating training triplets...")
    anchors, positives, negatives = create_triplet_data(person_images, num_triplets=2000)
    
    # Split into train/validation
    (anchor_train, anchor_val, positive_train, positive_val, 
     negative_train, negative_val) = train_test_split(
        anchors, positives, negatives, test_size=0.2, random_state=42
    )
    
    print(f"Training triplets: {len(anchor_train)}")
    print(f"Validation triplets: {len(anchor_val)}")
    
    # Create the embedding model
    print("Creating embedding model...")
    embedding_model = create_embedding_model(
        input_shape=(224, 224, 3), 
        embedding_dim=128
    )
    
    # Create the triplet model
    print("Creating triplet model...")
    triplet_model = create_triplet_model(embedding_model, input_shape=(224, 224, 3))
    
    # Compile the model
    triplet_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=triplet_loss(alpha=0.2)
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
    
    # Train the model
    print("Training the triplet model...")
    history = triplet_model.fit(
        [anchor_train, positive_train, negative_train],
        np.zeros(len(anchor_train)),  # Dummy labels for triplet loss
        validation_data=([anchor_val, positive_val, negative_val], np.zeros(len(anchor_val))),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save the trained embedding model
    print("Saving the embedding model...")
    embedding_model.save('face_embedding_model.keras')
    
    # Save person names for later use
    with open('person_names.pkl', 'wb') as f:
        pickle.dump(list(person_images.keys()), f)
    
    print("Training completed!")
    print("Model saved as 'face_embedding_model.keras'")
    print("Person names saved as 'person_names.pkl'")
    
    return embedding_model, person_images

def generate_embeddings_for_known_faces(embedding_model, person_images):
    """
    Generate embeddings for all known faces and save them.
    
    Parameters:
        embedding_model: Trained embedding model
        person_images (dict): Dictionary mapping person names to image lists
    """
    print("Generating embeddings for known faces...")
    
    known_embeddings = {}
    
    for person_name, images in person_images.items():
        embeddings = []
        for img in images:
            # Add batch dimension
            img_batch = np.expand_dims(img, axis=0)
            
            # Generate embedding
            embedding = embedding_model.predict(img_batch, verbose=0)
            embeddings.append(embedding.flatten())
        
        # Store average embedding for each person
        known_embeddings[person_name] = np.mean(embeddings, axis=0)
    
    # Save embeddings
    with open('known_face_embeddings.pkl', 'wb') as f:
        pickle.dump(known_embeddings, f)
    
    print(f"Saved embeddings for {len(known_embeddings)} people")
    return known_embeddings

if __name__ == "__main__":
    # Train the model
    embedding_model, person_images = train_embedding_model('resized_images')
    
    if embedding_model is not None:
        # Generate embeddings for known faces
        known_embeddings = generate_embeddings_for_known_faces(embedding_model, person_images)
