from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent

model_path = (
    SCRIPT_DIR / "runs/detect/WasteDetection/yolo26_waste_run_2/weights/best.pt"
)


def main():
    model = YOLO(model_path)
    metrics = model.val(split="test")
    print(metrics.box.maps)


if __name__ == "__main__":
    main()
