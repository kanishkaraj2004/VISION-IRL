import os
import sys
import tensorflow as tf
import cv2
import numpy as np
import pyttsx3  # ✅ NEW: for voice output

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Class labels (COCO 80 classes)
class_names = [
    "person", "laptop", "smartphone", "tablet", "mouse", "keyboard", "monitor",
    "desk", "chair", "whiteboard", "projector", "screen", "tv", "camera", "tripod",
    "microphone", "speaker", "headphones", "charger", "USB stick", "power strip",
    "extension board", "router", "network switch", "ethernet cable", "hdmi cable",
    "backpack", "bag", "notebook", "pen", "marker", "notepad", "clock", "door",
    "window", "curtain", "fan", "air conditioner", "light", "lamp", "plant",
    "bottle", "water bottle", "coffee cup", "coffee machine", "cup", "snack",
    "banana", "pizza box", "sandwich", "bowl", "plate", "spoon", "fork", "napkin",
    "mask", "hand sanitizer", "badge", "lanyard", "poster", "sign", "floor mat",
    "shoe", "jacket", "stool", "bench", "sofa", "refrigerator", "microwave",
    "toaster", "sink", "mirror", "trash bin", "recycle bin", "scanner", "printer",
    "clipboard", "tissue box", "vase", "umbrella", "glasses", "watch", "key", "book"
]

from utils.yolo_utils import preprocess, postprocess, draw_boxes



# ✅ NEW: Load SavedModel from TF format
def load_tf_model(model_path):
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return None
    try:
        model = tf.saved_model.load(model_path)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print("❌ Failed to load model:", e)
        return None

# ✅ Load the TensorFlow model
model_path = "models/##yolov5_saved_model"
model = load_tf_model(model_path)
if model is None:
    sys.exit("Exiting: Could not load model.")

# ✅ Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ✅ Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Error: Could not open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    input_tensor = preprocess(frame)

    try:
        infer = model.signatures["serving_default"]
        output_dict = infer(tf.convert_to_tensor(input_tensor))
        output_tensor = list(output_dict.values())[0]
    except Exception as e:
        print("❌ Model inference failed:", e)
        break

    original_shape = frame.shape[:2]
    boxes, scores, classes = postprocess(output_tensor, original_shape)
    frame = draw_boxes(frame, boxes, scores, classes, class_names)

    # ✅ Speak detected objects (once per frame)
    detected_objects = set([class_names[i] for i in classes])
    for obj in detected_objects:
        engine.say(f"{obj} ahead")
    engine.runAndWait()

    cv2.imshow("YOLO Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
