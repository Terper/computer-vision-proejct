from pathlib import Path
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = str(
    BASE_DIR
    / "model-training/runs/detect/WasteDetection/yolo26_waste_run_3/weights/best.pt"
)
model = YOLO(model_path)
model.to("cuda")


def load_images_from_folder(folder_path):
    images = []
    image_paths = []

    folder = Path(folder_path)

    for image_file in folder.iterdir():
        img = Image.open(image_file)
        images.append(img)
        image_paths.append(str(image_file))

    return images, image_paths


if __name__ == "__main__":
    folder_path = str(BASE_DIR / "datasets/Torture")
    images, paths = load_images_from_folder(folder_path)

    for img, path in zip(images, paths):
        result = model(img)
        result[0].save(f"output/{Path(path).stem}_detected.jpg")
