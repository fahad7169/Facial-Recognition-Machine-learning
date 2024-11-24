#model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_cnn_model(input_shape=(224, 224, 1)):
    """
    Creates a convolutional neural network (CNN) model for face recognition.

    Parameters:
        input_shape (tuple of int): Input shape of the model. Defaults to (224, 224, 1).

    Returns:
        keras.Model: The constructed CNN model.
    """

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
