import cv2
import numpy as np
import tensorflow as tf

def preprocess(frame, input_size=(640, 640)):
    """Preprocess frame for YOLO input"""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_size)
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

def load_tf_model(model_path):
    """Load TensorFlow SavedModel"""
    model = tf.saved_model.load(model_path)
    return model


def postprocess(predictions, original_shape, conf_threshold=0.5, iou_threshold=0.5):
    output = predictions.numpy()  # remove the [0] indexing here
    # output shape: [1, num_boxes, 85]

    boxes = []
    scores = []
    classes = []

    for detection in output[0]:  # access the first batch
        confidence = detection[4]
        if confidence < conf_threshold:
            continue

        x, y, w, h = detection[0:4]
        # ... rest remains same

        x1 = int((x - w/2) * original_shape[1])
        y1 = int((y - h/2) * original_shape[0])
        x2 = int((x + w/2) * original_shape[1])
        y2 = int((y + h/2) * original_shape[0])
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(original_shape[1]-1, x2), min(original_shape[0]-1, y2)
        
        class_scores = detection[5:]
        class_id = np.argmax(class_scores)
        score = float(confidence * class_scores[class_id])
        
        if score >= conf_threshold:
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            classes.append(int(class_id))
    
    if len(boxes) > 0:
        indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size=50,
            iou_threshold=iou_threshold, score_threshold=conf_threshold
        ).numpy()
        
        boxes = [boxes[i] for i in indices]
        scores = [scores[i] for i in indices]
        classes = [classes[i] for i in indices]
    
    return boxes, scores, classes

def draw_boxes(frame, boxes, scores, classes, class_names):
    """Draw bounding boxes on frame"""
    for box, score, class_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        color = (0, 255, 0)
        label = f"{class_names[class_id]}: {score:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame