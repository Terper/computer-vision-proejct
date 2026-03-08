from pathlib import Path
from PIL import Image
from ultralytics import YOLO

model = YOLO(
    "/Users/jann/Dev/Arcada/datorseende/project/model-training/runs/detect/WasteDetection/yolo26_waste_run_1/weights/best.pt"
)


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
    folder_path = "/Users/jann/Dev/Arcada/datorseende/project/datasets/torture-test"
    images, paths = load_images_from_folder(folder_path)

    for img, path in zip(images, paths):
        result = model(img)
        result[0].save(f"output/{Path(path).stem}_detected.jpg")
