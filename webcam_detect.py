import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run detection
    detection_result = detector.detect(mp_image)

    label = "Searching..."
    color = (255, 255, 255)

    if detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x, y = max(0, bbox.origin_x), max(0, bbox.origin_y)
            bw, bh = bbox.width, bbox.height

            # --- ADD PADDING (e.g., 20% expansion) ---
            pad_y = int(bh * 0.20)
            pad_x = int(bw * 0.20)
            
            y1 = max(0, y - pad_y)
            y2 = min(h, y + bh + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(w, x + bw + pad_x)

            # Crop the face with the new padded coordinates
            face_img = frame[y1:y2, x1:x2]

            if face_img.size > 0:
                # Preprocess
                resized_face = cv2.resize(face_img, (224, 224))
                rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
                input_data = np.expand_dims(rgb_face, axis=0)

                # Predict
                prediction = model.predict(input_data, verbose=0)[0][0]

                if prediction > 0.5:
                    label = "FATIGUE"
                    color = (0, 0, 255) # Red
                    confidence = prediction * 100
                else:
                    label = "ACTIVE"
                    color = (0, 255, 0) # Green
                    # If prediction is 0.2 (20% fatigue), it is 80% active
                    confidence = (1.0 - prediction) * 100 

                # Combine the label and the percentage (rounded to 1 decimal place)
                display_text = f"{label}: {confidence:.1f}%"

                # Draw the UI
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(frame, display_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Face-Focused Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()