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
        name="yolo26_waste_run_2",
        save=True,
        plots=True,
        augment=True,
        patience=50,
        lr0=0.01,
        cos_lr=True,
        mixup=0.1,
    )


if __name__ == "__main__":
    main()
