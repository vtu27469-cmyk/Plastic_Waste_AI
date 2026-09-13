from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

# Load ONNX model
net = cv2.dnn.readNetFromONNX(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image uploaded"
        }), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({
            "status": "error",
            "message": "No image selected"
        }), 400

    filename = str(uuid.uuid4()) + ".jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    image.save(image_path)

    try:

        frame = cv2.imread(image_path)

        if frame is None:
            return jsonify({
                "status": "error",
                "message": "Unable to read image"
            }), 400

        image_resized = cv2.resize(frame, (320, 320))

        blob = cv2.dnn.blobFromImage(
            image_resized,
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

        return jsonify({
            "status": "success",
            "plastic_detected": len(detected_objects) > 0,
            "plastic_count": len(detected_objects),
            "confidence": round(average_confidence, 2),
            "objects": detected_objects
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:

        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )