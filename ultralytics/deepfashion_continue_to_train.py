from ultralytics import YOLO
from glob import glob

# Load a pretrained YOLO11n model
model = YOLO("ultralytics/runs/deepfashion_exp/weights/last.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(resume=True,)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
image_paths = glob("~/SmartScan/datasets/deepfashion_yolo/images/test/img/*/*.jpg")
results = model(image_paths)
results[0].show()  # Display results