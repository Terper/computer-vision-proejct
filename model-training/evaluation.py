from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent

model_path = (
    SCRIPT_DIR / "runs/detect/WasteDetection/yolo26_waste_run_3/weights/best.pt"
)


def main():
    model = YOLO(model_path)
    model.to("cuda")
    model.val(
        split="test",
        save=True,
        plots=True,
        project="WasteDetection",
        name="test_results",
    )


if __name__ == "__main__":
    main()
