from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load trained plastic detection model
model = YOLO("runs/detect/plastic_waste_model/weights/best.pt")


@app.route("/")
def home():
    return "Plastic Waste Detection API is running!"


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]

    os.makedirs("uploads", exist_ok=True)
    image_path = os.path.join("uploads", image.filename)
    image.save(image_path)

    results = model.predict(
        source=image_path,
        conf=0.25
    )

    detections = []

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": model.names[class_id],
                "confidence": round(confidence, 2),
                "box": [
                    round(x1),
                    round(y1),
                    round(x2),
                    round(y2)
                ]
            })

    return jsonify({
        "count": len(detections),
        "detections": detections
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )