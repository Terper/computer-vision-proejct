import shutil
import splitfolders
import os
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


dataset_name = "TrashDataset"
dataset_dir = BASE_DIR / "datasets" / dataset_name

split_dir_name = dataset_dir.name + "-split"
split_dir = os.path.join(os.path.dirname(dataset_dir), split_dir_name)

if os.path.exists(split_dir):
    print(f"Split directory '{split_dir}' already exists.")
    exit(1)

cleaned_dir_name = dataset_dir.name + "-cleaned"
cleaned_dir = os.path.join(os.path.dirname(dataset_dir), cleaned_dir_name)
image_dir = os.path.join(cleaned_dir, "images")
label_dir = os.path.join(cleaned_dir, "labels")

os.makedirs(cleaned_dir, exist_ok=True)
shutil.copytree(dataset_dir, cleaned_dir, dirs_exist_ok=True)

image_files = {
    os.path.splitext(f)[0]: f for f in os.listdir(image_dir) if not f.startswith(".")
}
label_files = {
    os.path.splitext(f)[0]: f for f in os.listdir(label_dir) if not f.startswith(".")
}

common_stems = set(image_files.keys()) & set(label_files.keys())

images_deleted = 0
for stem, filename in image_files.items():
    if stem not in common_stems:
        os.remove(os.path.join(image_dir, filename))
        images_deleted += 1

labels_deleted = 0
for stem, filename in label_files.items():
    if stem not in common_stems:
        os.remove(os.path.join(label_dir, filename))
        labels_deleted += 1

print(f"Cleanup complete!")
print(f"Images deleted: {images_deleted}")
print(f"Labels deleted: {labels_deleted}")

splitfolders.ratio(
    cleaned_dir,
    output=split_dir,
    seed=random.randint(1, 10000),
    ratio=(0.8, 0.1, 0.1),
    shuffle=True,
    group="stem",
    move=True,
)

shutil.rmtree(cleaned_dir)
