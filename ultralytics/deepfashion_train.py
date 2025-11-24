import shutil
import time
from glob import glob
from pathlib import Path

from ultralytics import YOLO


# Custom callback để lưu checkpoint lên Google Drive theo thời gian
class DriveTimeBasedCheckpoint:
    def __init__(self, save_interval_minutes=10, drive_path="/content/drive/MyDrive/yolo_checkpoints"):
        self.save_interval = save_interval_minutes * 60  # Convert to seconds
        self.last_save_time = time.time()
        self.drive_path = Path(drive_path)
        self.local_checkpoint_dir = None

        # Create drive directory if not exists
        self.drive_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Drive checkpoint directory ready: {self.drive_path}")

    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch."""
        current_time = time.time()
        elapsed = current_time - self.last_save_time

        if elapsed >= self.save_interval:
            # Set local checkpoint directory if not set
            if self.local_checkpoint_dir is None:
                self.local_checkpoint_dir = Path(trainer.save_dir) / "time_checkpoints"
                self.local_checkpoint_dir.mkdir(exist_ok=True)

            # Save checkpoint locally first
            epoch = trainer.epoch
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_epoch{epoch}_{timestamp}.pt"
            local_checkpoint_path = self.local_checkpoint_dir / checkpoint_name

            # Save model
            trainer.save_model(local_checkpoint_path)

            # Copy to Google Drive
            drive_checkpoint_path = self.drive_path / checkpoint_name
            try:
                shutil.copy2(local_checkpoint_path, drive_checkpoint_path)
                print(f"\n⏰ Checkpoint saved to Drive: {drive_checkpoint_path}")
                print(f"   Epoch: {epoch} | Elapsed: {elapsed / 60:.1f} minutes\n")
            except Exception as e:
                print(f"\n❌ Error copying to Drive: {e}\n")

            # Update last save time
            self.last_save_time = current_time


# Mount Google Drive (nếu chạy trên Colab)
try:
    from google.colab import drive

    drive.mount("/content/drive")
    print("✅ Google Drive mounted successfully")
except:
    print("ℹ️  Not running on Colab or Drive already mounted")

# Load a pretrained YOLO11n model
model = YOLO("yolo11x.pt")

# Create callback instance - Thay đổi drive_path theo ý bạn
drive_checkpoint = DriveTimeBasedCheckpoint(
    save_interval_minutes=10,
    drive_path="/content/drive/MyDrive/yolo_checkpoints",  # Đường dẫn Drive
)

# Add callback to model
model.add_callback("on_train_epoch_end", drive_checkpoint.on_train_epoch_end)

# Train the model
train_results = model.train(
    data="datasets/deepfashion_yolo/deepfashion.yaml",
    fraction=0.10,
    epochs=10,
    batch=2,
    imgsz=640,
    device="0",
    workers=2,
    patience=3,
    save=True,
    exist_ok=False,
    pretrained=True,
    optimizer="auto",
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
    fliplr=0.5,
    mosaic=0.0,
    mixup=0.0,
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
image_paths = glob("~/SmartScan/datasets/deepfashion_yolo/images/test/img/*/*.jpg")
results = model(image_paths)
results[0].show()
