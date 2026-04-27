import cv2
import tensorflow as tf
import numpy as np

# 1. Load your trained model
model = tf.keras.models.load_model('drowsy_model_v1.keras')

# 2. Start the Webcam (0 is usually the default laptop camera)
cap = cv2.VideoCapture(0)

print("Starting Webcam... Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # --- Preprocessing for the Model ---
    # 1. Flip frame horizontally for a "mirror" effect (optional)
    frame = cv2.flip(frame, 1)
    
    # 2. Convert BGR (OpenCV default) to RGB (Model default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 3. Resize to 224x224
    resized_frame = cv2.resize(rgb_frame, (224, 224))
    
    # 4. Add batch dimension (1, 224, 224, 3)
    input_data = np.expand_dims(resized_frame, axis=0)

    # --- Prediction ---
    prediction = model.predict(input_data, verbose=0)[0][0]
    
    # --- Logic & Visualization ---
    if prediction > 0.5:
        label = "FATIGUE"
        color = (0, 0, 255) # Red in BGR
        confidence = prediction
    else:
        label = "ACTIVE"
        color = (0, 255, 0) # Green in BGR
        confidence = 1 - prediction

    # Display the result on the frame
    text = f"{label}: {confidence*100:.1f}%"
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Draw a border/rectangle if Fatigue is detected
    if label == "FATIGUE":
        cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 10)

    # Show the video feed
    cv2.imshow('Drowsiness Detection System', frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()