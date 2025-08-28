#model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Lambda
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

def create_embedding_model(input_shape=(224, 224, 3), embedding_dim=128):
    """
    Creates a deep embedding model for face recognition.
    This model generates compact numerical representations (embeddings) of faces.
    
    Parameters:
        input_shape (tuple): Input shape (height, width, channels)
        embedding_dim (int): Dimension of the output embedding vector
    
    Returns:
        keras.Model: The embedding model
    """
    
    # Use ResNet50 as base model (pre-trained on ImageNet)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom layers for face embedding
    x = base_model.output
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output embedding layer
    embedding = Dense(embedding_dim, activation=None, name='embedding')(x)
    
    # L2 normalization for the embedding
    embedding = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_normalization')(embedding)
    
    model = Model(inputs=base_model.input, outputs=embedding)
    
    return model

def create_siamese_model(input_shape=(224, 224, 3), embedding_dim=128):
    """
    Creates a Siamese network for training face embeddings.
    
    Parameters:
        input_shape (tuple): Input shape (height, width, channels)
        embedding_dim (int): Dimension of the output embedding vector
    
    Returns:
        keras.Model: The Siamese model
    """
    
    # Create the embedding model
    embedding_model = create_embedding_model(input_shape, embedding_dim)
    
    # Create two inputs for the Siamese network
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    # Generate embeddings for both inputs
    embedding_a = embedding_model(input_a)
    embedding_b = embedding_model(input_b)
    
    # Calculate the distance between embeddings
    distance = Lambda(lambda x: tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True))([embedding_a, embedding_b])
    
    # Create the Siamese model
    siamese_model = Model(inputs=[input_a, input_b], outputs=distance)
    
    return siamese_model, embedding_model

def triplet_loss(alpha=0.2):
    """
    Triplet loss function for training face embeddings.
    
    Parameters:
        alpha (float): Margin parameter for the triplet loss
    
    Returns:
        function: Triplet loss function
    """
    
    def loss(y_true, y_pred):
        # y_true: 1 for positive pairs, 0 for negative pairs
        # y_pred: distance between embeddings
        
        positive_dist = y_pred
        negative_dist = y_pred
        
        # Triplet loss: max(positive_dist - negative_dist + alpha, 0)
        basic_loss = positive_dist - negative_dist + alpha
        loss = tf.maximum(basic_loss, 0.0)
        
        return tf.reduce_mean(loss)
    
    return loss

def create_triplet_model(input_shape=(224, 224, 3), embedding_dim=128):
    """
    Creates a triplet model for training face embeddings.
    
    Parameters:
        input_shape (tuple): Input shape (height, width, channels)
        embedding_dim (int): Dimension of the output embedding vector
    
    Returns:
        keras.Model: The triplet model
    """
    
    # Create the embedding model
    embedding_model = create_embedding_model(input_shape, embedding_dim)
    
    # Create three inputs for anchor, positive, and negative
    anchor_input = Input(shape=input_shape, name='anchor_input')
    positive_input = Input(shape=input_shape, name='positive_input')
    negative_input = Input(shape=input_shape, name='negative_input')
    
    # Generate embeddings for all three inputs
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)
    
    # Create the triplet model
    triplet_model = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=[anchor_embedding, positive_embedding, negative_embedding]
    )
    
    return triplet_model, embedding_model

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    
    Parameters:
        a, b: numpy arrays or tensors
    
    Returns:
        float: Cosine similarity score
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    """
    Calculate Euclidean distance between two vectors.
    
    Parameters:
        a, b: numpy arrays or tensors
    
    Returns:
        float: Euclidean distance
    """
    return np.linalg.norm(a - b)
