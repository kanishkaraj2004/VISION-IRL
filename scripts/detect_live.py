import os
import sys
import tensorflow as tf
import cv2
import numpy as np

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

# Import helper functions
from utils.yolo_utils import preprocess, postprocess, draw_boxes

# ✅ NEW: Define load_tf_model here since it's missing from yolo_utils
def load_tf_model(model_path):
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return None

    try:
        files = os.listdir(model_path)
        print(f"Contents of model directory: {files}")
        model = tf.saved_model.load(model_path)  # Will work if valid SavedModel format
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print("❌ Failed to load model:", e)
        return None


# ✅ PATH to SavedModel directory (must contain saved_model.pb + variables/)
model_path = "models/yolov5/yolov5s_saved_model"  # Change if different
model = load_tf_model(model_path)

# 🧪 Quick test of model before running live detection
if model is None:
    print("Exiting because model could not be loaded.")
    sys.exit(1)

# --- Live webcam detection code (simplified demo) ---

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    input_tensor = preprocess(frame)

    try:
        infer = model.signatures["serving_default"]
        output_dict = infer(tf.convert_to_tensor(input_tensor))
        # FIXED: get the first tensor from output_dict (avoid KeyError 0)
        output_tensor = list(output_dict.values())[0]
    except Exception as e:
        print("❌ Model inference failed:", e)
        break

    original_shape = frame.shape[:2]
    print("output_tensor:", output_tensor)
    print("output_tensor shape:", output_tensor.shape)

    boxes, scores, classes = postprocess(output_tensor, original_shape)
    frame = draw_boxes(frame, boxes, scores, classes, class_names)

    cv2.imshow("YOLO Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
