import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")

CLASS_NAMES = [
    "Black Plastic Cap",
    "Blue Nitrile Glove",
    "Blue Plastic Cap",
    "Brown Multilayer Plastic",
    "Green Plastic Cap",
    "Orange Plastic Cap",
    "Plastic Bottle",
    "Purple Insulation Foam",
    "Purple Multilayer Plastic Bag",
    "Red-Orange BOPP Bag",
    "Red Cap",
    "Red Netting",
    "Red Plastic Straw",
    "Yellow Foam",
    "Yellow Rope"
]

net = cv2.dnn.readNetFromONNX(MODEL_PATH)


def detect_plastic(image_path):

    frame = cv2.imread(image_path)

    if frame is None:
        raise ValueError("Unable to read image")

    image = cv2.resize(frame, (320, 320))

    blob = cv2.dnn.blobFromImage(
        image,
        1 / 255.0,
        (320, 320),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)

    outputs = net.forward()

    output = outputs[0]

    if len(output.shape) == 3:
        output = output[0]

    if output.shape[0] < output.shape[1]:
        output = output.transpose()

    detected_objects = []
    confidence_values = []

    for detection in output:

        if len(detection) < 19:
            continue

        class_scores = detection[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < 0.25:
            continue

        if class_id >= len(CLASS_NAMES):
            continue

        detected_objects.append({
            "class": CLASS_NAMES[class_id],
            "confidence": round(confidence * 100, 2)
        })

        confidence_values.append(confidence)

    if confidence_values:
        average_confidence = (
            sum(confidence_values) /
            len(confidence_values)
        ) * 100
    else:
        average_confidence = 0

    return {
        "plastic_count": len(detected_objects),
        "detected_objects": detected_objects,
        "confidence": round(average_confidence, 2)
    }