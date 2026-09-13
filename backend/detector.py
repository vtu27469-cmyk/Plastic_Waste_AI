from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "plastic_waste_model",
    "weights",
    "best.pt"
)

model = YOLO(MODEL_PATH)


def detect_plastic(image_path):
    results = model(image_path, conf=0.25)

    detected_objects = []
    plastic_count = 0
    confidence_values = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            confidence_values.append(confidence)

            detected_objects.append({
                "class": class_name,
                "confidence": round(confidence * 100, 2)
            })

            plastic_count += 1

    if confidence_values:
        average_confidence = (
            sum(confidence_values) / len(confidence_values)
        ) * 100
    else:
        average_confidence = 0

    return {
        "plastic_count": plastic_count,
        "detected_objects": detected_objects,
        "confidence": round(average_confidence, 2)
    }