from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent

model = YOLO("yolo26s.pt")


def main():
    results = model.train(
        data=SCRIPT_DIR / "data.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        # device="mps",  # using apple silicon
        device="cuda",  # using gpu
        project="WasteDetection",
        name="yolo26_waste_run_1",
        save=True,
        plots=True,
        patience=50,
        cos_lr=True,
        mixup=0.1,
        flipud=0.1,
        copy_paste=0.1,
        degrees=10,
        shear=5,
        dropout=0.1,
        label_smoothing=0.1,
    )


if __name__ == "__main__":
    main()
