from ultralytics import YOLO

# Load trained plastic waste model
model = YOLO("runs/detect/plastic_waste_model/weights/best.pt")

# Test on dataset images
results = model.predict(
    source="dataset/plastic_in_water/test/images",
    conf=0.25,
    save=True
)

print("Plastic detection completed successfully!")