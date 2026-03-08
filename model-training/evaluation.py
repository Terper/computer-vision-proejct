from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent

model = YOLO(
    "/Users/jann/Dev/Arcada/datorseende/project/model-training/runs/detect/WasteDetection/yolo26_waste_run_1/weights/best.pt"
)


def main():
    metrics = model.val(split="test")
    print(metrics.box.maps)


if __name__ == "__main__":
    main()
