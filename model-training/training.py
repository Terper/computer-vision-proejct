from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent

model = YOLO("yolo26n.pt")


def main():
    results = model.train(
        data=SCRIPT_DIR / "data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="mps",  # using apple silicon
        project="WasteDetection",
        name="yolo26_waste_run_1",
        save=True,
        plots=True,
        augment=True,
    )


if __name__ == "__main__":
    main()
