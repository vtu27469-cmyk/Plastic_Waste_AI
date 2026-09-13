from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load trained plastic detection model
model = YOLO("best.pt")


@app.route("/")
def home():
    return "Plastic Waste Detection API is running!"


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "message": "Plastic Waste Detection API is running"
    })


@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({
            "error": "No image provided"
        }), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({
            "error": "No image selected"
        }), 400

    # Create uploads folder
    os.makedirs("uploads", exist_ok=True)

    image_path = os.path.join(
        "uploads",
        image.filename
    )

    # Save uploaded image
    image.save(image_path)

    try:

        # Run YOLO plastic detection
        results = model.predict(
            source=image_path,
            conf=0.25
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

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
            "status": "success",
            "plastic_detected": len(detections) > 0,
            "count": len(detections),
            "detections": detections
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

    finally:

        # Delete uploaded image after processing
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
