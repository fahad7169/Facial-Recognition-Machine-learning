# Advanced Face Recognition System

This is a redesigned face recognition system that uses **face embeddings** instead of traditional classification. The new approach provides much better accuracy and can distinguish between similar faces more effectively.

## 🚀 Key Improvements

### 1. **Embedding-Based Recognition**
- **Before**: Simple CNN classification (limited accuracy)
- **Now**: Deep face embeddings with similarity comparison
- **Result**: Much better face distinction and recognition accuracy

### 2. **Advanced Architecture**
- **ResNet50 backbone**: Pre-trained on ImageNet for robust feature extraction
- **Siamese Network**: Twin networks for learning face similarities
- **Triplet Loss**: Trains the model to push different faces apart and similar faces closer

### 3. **Better Preprocessing**
- **Face Alignment**: Uses facial landmarks for proper face orientation
- **Image Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **RGB Processing**: Full color information instead of grayscale

### 4. **Multiple Similarity Metrics**
- **Cosine Similarity**: Measures angular similarity between embeddings
- **Euclidean Distance**: Measures spatial distance between embeddings
- **Adaptive Selection**: Automatically chooses the better metric for each comparison

## 📁 Project Structure

```
Facial Recognition/
├── model.py              # New embedding model architecture
├── train.py              # Training script with triplet loss
├── predict.py            # Recognition system with embedding comparison
├── utils.py              # Face preprocessing and utility functions
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── resized_images/      # Your face dataset (person folders)
    ├── person1/
    ├── person2/
    └── ...
```

## 🛠️ Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Optional: Install dlib for facial landmarks**
   - **Windows**: `pip install dlib`
   - **Linux/Mac**: `conda install -c conda-forge dlib`

## 🎯 How It Works

### 1. **Training Phase**
```python
# The system learns to create face embeddings
python train.py
```

**What happens:**
- Loads face images from `resized_images/` folder
- Creates training triplets (anchor, positive, negative)
- Trains ResNet50-based embedding model using triplet loss
- Saves trained model and generates embeddings for known faces

### 2. **Recognition Phase**
```python
# Run real-time face recognition
python predict.py
```

**What happens:**
- Detects faces in webcam feed
- Generates embeddings for detected faces
- Compares with stored known face embeddings
- Uses similarity thresholds for recognition

## 🔧 Usage

### **Step 1: Prepare Your Dataset**
Organize your face images like this:
```
resized_images/
├── John_Doe/
│   ├── john1.jpg
│   ├── john2.jpg
│   └── john3.jpg
├── Jane_Smith/
│   ├── jane1.jpg
│   ├── jane2.jpg
│   └── jane3.jpg
└── ...
```

**Requirements:**
- At least 5 images per person
- Images should be clear, well-lit faces
- Supported formats: JPG, JPEG, PNG

### **Step 2: Train the Model**
```bash
python train.py
```

**Training Process:**
- Creates 2000 training triplets
- Uses 80/20 train/validation split
- Trains for up to 50 epochs with early stopping
- Saves model as `face_embedding_model.keras`

### **Step 3: Run Recognition**
```bash
python predict.py
```

**Controls:**
- Press `q` to quit
- Press `s` to save current frame

## 📊 Performance Metrics

The new system provides:
- **Higher Accuracy**: Better distinction between similar faces
- **Lower False Positives**: Reduced misidentifications
- **Robust Recognition**: Works with different lighting, angles, expressions
- **Scalability**: Easy to add new faces without retraining

## 🔍 Technical Details

### **Model Architecture**
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Embedding Dimension**: 128-dimensional vectors
- **Loss Function**: Triplet Loss with margin α = 0.2
- **Optimizer**: Adam with learning rate 0.0001

### **Similarity Thresholds**
- **Default Threshold**: 0.6 (60% similarity)
- **Adjustable**: Modify `similarity_threshold` in `FaceRecognitionSystem`
- **Methods**: Cosine similarity and Euclidean distance

### **Preprocessing Pipeline**
1. **Face Detection**: Haar Cascade classifier
2. **Landmark Detection**: 68 facial points (if dlib available)
3. **Face Alignment**: Eye-based rotation correction
4. **Image Enhancement**: CLAHE for better contrast
5. **Normalization**: Scale to [0, 1] range

## 🚨 Troubleshooting

### **Common Issues**

1. **"Model file not found"**
   - Make sure you've run `train.py` first
   - Check that `face_embedding_model.keras` exists

2. **"No known faces"**
   - Ensure `known_face_embeddings.pkl` was created during training
   - Verify your dataset has sufficient images per person

3. **Poor recognition accuracy**
   - Increase number of training images per person
   - Adjust similarity threshold in `predict.py`
   - Ensure good quality, well-lit face images

4. **dlib installation issues**
   - The system works without dlib (uses basic preprocessing)
   - For better results, try conda installation

### **Performance Tips**

1. **Dataset Quality**
   - Use high-resolution images (224x224 minimum)
   - Include various expressions and lighting conditions
   - Ensure faces are clearly visible and centered

2. **Training Parameters**
   - Increase `num_triplets` for better training
   - Adjust `min_images_per_person` based on your data
   - Monitor validation loss during training

3. **Recognition Settings**
   - Tune `similarity_threshold` for your use case
   - Lower threshold = more permissive recognition
   - Higher threshold = stricter recognition

## 🔮 Future Enhancements

- **Face Quality Assessment**: Automatic filtering of poor quality images
- **Online Learning**: Add new faces without full retraining
- **Multi-Face Tracking**: Handle multiple faces simultaneously
- **Emotion Recognition**: Detect facial expressions
- **Age/Gender Estimation**: Additional demographic information

## 📝 License

This project is open source. Feel free to modify and distribute according to your needs.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

**Note**: This system requires a reasonable amount of training data (5+ images per person) for optimal performance. The more diverse and high-quality your training images are, the better the recognition accuracy will be.
