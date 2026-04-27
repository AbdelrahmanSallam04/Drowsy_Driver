import tensorflow as tf
import cv2
import numpy as np

# 1. Load the saved model
# Note: Keras handles the custom Rescaling layer automatically
model = tf.keras.models.load_model('drowsy_model_v1.keras')

def prepare_image(image_path):
    """Loads an image and prepares it for the model."""
    img_size = (224, 224)
    
    # Load using OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    
    # Resize to match the model's expected input
    img = cv2.resize(img, img_size)
    
    # Expand dimensions to create a batch (1, 224, 224, 3)
    img_array = np.expand_dims(img, axis=0)
    
    return img_array

def run_prediction(image_path):
    # Preprocess
    processed_img = prepare_image(image_path)
    
    # Predict
    # The output is a probability between 0 and 1
    prediction = model.predict(processed_img)
    
    # Map to classes (0 = active, 1 = fatigue based on folder order)
    if prediction[0][0] > 0.5:
        label = "Fatigue"
        confidence = prediction[0][0]
    else:
        label = "Active"
        confidence = 1 - prediction[0][0]
        
    print(f"Prediction: {label} ({confidence*100:.2f}%)")

# Test it
run_prediction('test/image__870.jpg')
run_prediction('test/image__1015.jpg')