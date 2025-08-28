#test_system.py
import os
import cv2
import numpy as np
import pickle
from model import create_embedding_model, cosine_similarity, euclidean_distance
from utils import FacePreprocessor, calculate_embedding_similarity
import matplotlib.pyplot as plt

def test_embedding_model():
    """Test the embedding model creation and basic functionality."""
    print("Testing embedding model creation...")
    
    try:
        # Create the embedding model
        model = create_embedding_model(input_shape=(224, 224, 3), embedding_dim=128)
        
        # Test with dummy input
        dummy_input = np.random.random((1, 224, 224, 3))
        embedding = model.predict(dummy_input, verbose=0)
        
        print(f"✓ Model created successfully")
        print(f"✓ Input shape: {dummy_input.shape}")
        print(f"✓ Output embedding shape: {embedding.shape}")
        print(f"✓ Embedding dimension: {embedding.shape[-1]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False

def test_face_preprocessor():
    """Test the face preprocessor functionality."""
    print("\nTesting face preprocessor...")
    
    try:
        # Create preprocessor
        preprocessor = FacePreprocessor()
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        # Test preprocessing
        processed = preprocessor.preprocess_for_embedding(
            dummy_image, 
            target_size=(224, 224),
            use_landmarks=False,  # Don't use landmarks for this test
            enhance=True
        )
        
        print(f"✓ Preprocessor created successfully")
        print(f"✓ Input image shape: {dummy_image.shape}")
        print(f"✓ Processed image shape: {processed.shape}")
        print(f"✓ Processed image range: [{processed.min():.3f}, {processed.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing preprocessor: {e}")
        return False

def test_similarity_functions():
    """Test similarity calculation functions."""
    print("\nTesting similarity functions...")
    
    try:
        # Create dummy embeddings
        embedding1 = np.random.random(128)
        embedding2 = np.random.random(128)
        
        # Test cosine similarity
        cosine_sim = cosine_similarity(embedding1, embedding2)
        
        # Test euclidean distance
        euclidean_dist = euclidean_distance(embedding1, embedding2)
        
        # Test utility function
        util_cosine = calculate_embedding_similarity(embedding1, embedding2, method='cosine')
        util_euclidean = calculate_embedding_similarity(embedding1, embedding2, method='euclidean')
        
        print(f"✓ Cosine similarity: {cosine_sim:.4f}")
        print(f"✓ Euclidean distance: {euclidean_dist:.4f}")
        print(f"✓ Utility cosine: {util_cosine:.4f}")
        print(f"✓ Utility euclidean: {util_euclidean:.4f}")
        
        # Verify consistency
        assert abs(cosine_sim - util_cosine) < 1e-6, "Cosine similarity mismatch"
        print(f"✓ All similarity functions working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing similarity functions: {e}")
        return False

def test_model_files():
    """Check if required model files exist."""
    print("\nChecking model files...")
    
    required_files = [
        'face_embedding_model.keras',
        'known_face_embeddings.pkl',
        'person_names.pkl'
    ]
    
    missing_files = []
    existing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            existing_files.append(file)
            file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
            print(f"✓ {file} exists ({file_size:.2f} MB)")
        else:
            missing_files.append(file)
            print(f"✗ {file} missing")
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        print("   Run 'python train.py' to create these files")
        return False
    else:
        print(f"\n✓ All required model files found")
        return True

def test_loaded_model():
    """Test the loaded model if it exists."""
    print("\nTesting loaded model...")
    
    try:
        from tensorflow.keras.models import load_model
        
        if not os.path.exists('face_embedding_model.keras'):
            print("⚠️  Model file not found, skipping test")
            return False
        
        # Load the model
        model = load_model('face_embedding_model.keras', compile=False)
        
        # Test with dummy input
        dummy_input = np.random.random((1, 224, 224, 3))
        embedding = model.predict(dummy_input, verbose=0)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Model input shape: {model.input_shape}")
        print(f"✓ Model output shape: {model.output_shape}")
        print(f"✓ Test embedding shape: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing loaded model: {e}")
        return False

def test_embeddings():
    """Test the loaded embeddings if they exist."""
    print("\nTesting loaded embeddings...")
    
    try:
        if not os.path.exists('known_face_embeddings.pkl'):
            print("⚠️  Embeddings file not found, skipping test")
            return False
        
        # Load embeddings
        with open('known_face_embeddings.pkl', 'rb') as f:
            known_embeddings = pickle.load(f)
        
        print(f"✓ Embeddings loaded successfully")
        print(f"✓ Number of known faces: {len(known_embeddings)}")
        
        # Test a few embeddings
        for i, (person_name, embedding) in enumerate(list(known_embeddings.items())[:3]):
            print(f"  - {person_name}: embedding shape {embedding.shape}")
        
        if len(known_embeddings) > 3:
            print(f"  ... and {len(known_embeddings) - 3} more")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing embeddings: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("=" * 60)
    print("FACE RECOGNITION SYSTEM COMPREHENSIVE TEST")
    print("=" * 60)
    
    tests = [
        ("Embedding Model Creation", test_embedding_model),
        ("Face Preprocessor", test_face_preprocessor),
        ("Similarity Functions", test_similarity_functions),
        ("Model Files Check", test_model_files),
        ("Loaded Model Test", test_loaded_model),
        ("Embeddings Test", test_embeddings)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    run_comprehensive_test()
