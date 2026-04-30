import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. Download the face detection model if not present
MODEL_PATH = "blaze_face_short_range.tflite"
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    print("Downloading MediaPipe face model...")
    urllib.request.urlretrieve(url, MODEL_PATH)

# 2. Initialize the new MediaPipe Face Detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)
detector = vision.FaceDetector.create_from_options(options)

# 3. Load your Keras model
model = tf.keras.models.load_model('drowsy_model_v1.keras')

# --- Smoothing Buffer ---
prediction_history = deque(maxlen=10) 

# Calibration variables
IS_CALIBRATING = True
CALIBRATION_FRAMES = 60  # ~2 seconds of video at 30fps
baseline_scores = []
baseline_avg = 0.5 # Stores the moving average
ALPHA = 0.01 # Controls adaptation speed (0.01 means 1% new data, 99% old)
personalized_threshold = 0.50 # Default fallback

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    if detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            # Square padding is better for MobileNet
            side = max(bw, bh)
            pad = int(side * 0.25)
            
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + side + pad)
            y2 = min(h, y + side + pad)

            face_img = frame[y1:y2, x1:x2]

            if face_img.size > 0:
                # Preprocessing
                resized_face = cv2.resize(face_img, (224, 224))
                # Convert to float32 for faster inference
                input_data = np.expand_dims(resized_face, axis=0).astype(np.float32)

                # --- SPEED FIX: Call model directly instead of .predict() ---
                preds = model(input_data, training=False)
                conf = preds.numpy()[0][0]
                prediction_history.append(conf)

                # --- 3. THE CALIBRATION MECHANISM ---
                if IS_CALIBRATING:
                    baseline_scores.append(conf)
                    
                    # Draw Calibration UI
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2) # Orange
                    cv2.putText(frame, f"CALIBRATING... {len(baseline_scores)}/{CALIBRATION_FRAMES}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    cv2.putText(frame, "Keep eyes normally open!", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # Finish Calibration
                    if len(baseline_scores) >= CALIBRATION_FRAMES:
                        # Find their average "Active" score
                        baseline_avg = np.mean(baseline_scores)
                        
                        # Set threshold to 15% higher than their natural resting state
                        # We cap it at 0.85 so it's not impossible to trigger
                        personalized_threshold = min(0.85, baseline_avg + 0.15)
                        
                        print(f"Calibration Complete. Baseline: {baseline_avg:.2f}, New Threshold: {personalized_threshold:.2f}")
                        IS_CALIBRATING = False
                else:
                    # --- LOGIC: Average the last 10 frames ---
                    avg_conf = sum(prediction_history) / len(prediction_history)

                    if avg_conf > personalized_threshold:
                        label, color = "FATIGUE", (0, 0, 255)
                        display_conf = avg_conf * 100
                    else:
                        label, color = "ACTIVE", (0, 255, 0)
                        # --- ADAPTIVE MOVING AVERAGE ---
                        # Only update the baseline while the person is ACTIVE
                        # This allows the model to adjust to new lighting or slight head tilts
                        baseline_avg = (ALPHA * avg_conf) + (1 - ALPHA) * baseline_avg
                        personalized_threshold = min(0.85, baseline_avg + 0.15)

                        # Recalculate display percentage relative to their threshold so UI makes sense
                        display_conf = (1 - avg_conf) * 100 
                        display_conf = max(0, display_conf) # Prevent negative numbers

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label}: {display_conf:.1f}%", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Debug text to see the active threshold
                    cv2.putText(frame, f"Threshold: {personalized_threshold:.2f}", (10, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Drowsiness Detection', frame)
    # Press 'c' to manually trigger recalibration
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('c'):
        IS_CALIBRATING = True
        baseline_scores = []

cap.release()
cv2.destroyAllWindows()