from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained plastic detection model
model = YOLO(MODEL_PATH)


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

    filename = str(uuid.uuid4()) + "_" + image.filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    image.save(image_path)

    try:

        results = model.predict(
            source=image_path,
            conf=0.25,
            verbose=False
        )

        detected_objects = []
        confidence_values = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                detected_objects.append({
                    "class": class_name,
                    "confidence": round(confidence * 100, 2)
                })

                confidence_values.append(confidence)

        plastic_count = len(detected_objects)

        if confidence_values:
            average_confidence = (
                sum(confidence_values) /
                len(confidence_values)
            ) * 100
        else:
            average_confidence = 0

        return jsonify({
            "status": "success",
            "plastic_detected": plastic_count > 0,
            "plastic_count": plastic_count,
            "detected_objects": detected_objects,
            "confidence": round(average_confidence, 2)
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
