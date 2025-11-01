from ultralytics import YOLO
from glob import glob

# Load a pretrained YOLO11n model
model = YOLO("yolo11x.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="datasets/deepfashion_yolo/deepfashion.yaml",  # Path to dataset configuration file
    fraction=0.10,
    epochs=10,  # Number of training epochs
    batch=2,
    imgsz=640,  # Image size for training
    device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    workers=2,
    patience=3,            # Early stopping
    save=True,
    exist_ok=False,
    pretrained=True,
    optimizer='auto',
    verbose=True,
    seed=42,
    deterministic=True,
    single_cls=True,
    val=False,
    plots=False,
    lr0=0.02,
    lrf=0.001,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=1.0,
    
    # Loss weights
    box=7.5,
    cls=0.5,
    dfl=1.5,
    
    # Augmentation (conservative for fashion)
    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=0.0,
    translate=0.1,
    scale=0.3,
    flipud=0.0,
    fliplr=0.5,            # Horizontal flip OK for clothes
    mosaic=0.0,
    mixup=0.0,

)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
image_paths = glob("~/SmartScan/datasets/deepfashion_yolo/images/test/img/*/*.jpg")
results = model(image_paths)
results[0].show()  # Display results