from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import os
import uuid
import json

from database import (
    initialize_database,
    save_detection,
    get_all_detections
)

from detector import detect_plastic


# Project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Create Flask application
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

CORS(app)


# Upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Initialize database
initialize_database()


# Home page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })


# Plastic detection
@app.route("/detect", methods=["POST"])
def detect():

    try:

        if "image" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image uploaded"
            }), 400

        image = request.files["image"]

        if image.filename == "":
            return jsonify({
                "status": "error",
                "message": "Invalid image filename"
            }), 400

        extension = os.path.splitext(image.filename)[1]

        filename = str(uuid.uuid4()) + extension

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        image.save(image_path)


        # Run YOLO plastic detection
        result = detect_plastic(image_path)


        # Convert detected objects to JSON
        detected_objects_json = json.dumps(
            result["detected_objects"]
        )


        # Save result to database
        detection_id = save_detection(
            filename,
            result["plastic_count"],
            detected_objects_json,
            result["confidence"]
        )


        # Delete uploaded image after detection
        if os.path.exists(image_path):
            os.remove(image_path)


        # Send result to web app
        return jsonify({
            "status": "success",
            "detection_id": detection_id,
            "image_name": filename,
            "plastic_detected": result["plastic_count"] > 0,
            "plastic_count": result["plastic_count"],
            "confidence": result["confidence"],
            "objects": result["detected_objects"]
        }), 200


    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Detection history
@app.route("/history", methods=["GET"])
def history():

    try:

        detections = get_all_detections()

        return jsonify({
            "status": "success",
            "count": len(detections),
            "history": detections
        }), 200


    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Get one detection
@app.route("/history/<int:detection_id>", methods=["GET"])
def single_detection(detection_id):

    try:

        detections = get_all_detections()

        for detection in detections:

            if detection["id"] == detection_id:

                return jsonify({
                    "status": "success",
                    "data": detection
                }), 200


        return jsonify({
            "status": "error",
            "message": "Detection not found"
        }), 404


    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Start server
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )