import math
import numpy as np
from itertools import combinations
from pathlib import Path
import cv2
from ultralytics import YOLO

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2
BOX_THICKNESS = 5
LINE_THICKNESS = 2
BOX_PADDING = 10
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
CLUSTER_THRESHOLD = 0.2

DEBUG = False

PLURAL_LABELS = {
    "Dryckeskartong": "Dryckeskartonger",
    "Konservburk": "Konservburkar",
    "Pantburk": "Pantburkar",
}


BASE_DIR = Path(__file__).parent
input_path = str(BASE_DIR / "inputdd.mp4")
output_path = str(BASE_DIR / "output.mp4")

model_path = str(
    BASE_DIR.parent
    / "model-training/runs/detect/WasteDetection/yolo26_waste_run_3/weights/best.pt"
)
model = YOLO(model_path)
model.to("cuda")


def get_centroid(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def apply_padding(box, frame_shape, padding=BOX_PADDING):
    x1, y1, x2, y2 = map(int, box)
    height, width = frame_shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return x1, y1, x2, y2


def draw_box(frame, box, color, padding=BOX_PADDING):
    x1, y1, x2, y2 = apply_padding(box, frame.shape, padding)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)


def draw_label(frame, box, label, color, padding=BOX_PADDING):
    x1, y1, x2, _ = apply_padding(box, frame.shape, padding)
    label_text = str(label)
    (text_w, text_h), baseline = cv2.getTextSize(
        label_text, FONT, FONT_SCALE, FONT_THICKNESS
    )
    box_center_x = (x1 + x2) // 2
    text_x = max(0, min(frame.shape[1] - text_w, box_center_x - text_w // 2))
    text_y = max(text_h + baseline, y1 - 10 - BOX_THICKNESS)

    cv2.putText(
        frame,
        label_text,
        (text_x, text_y),
        FONT,
        FONT_SCALE,
        color,
        FONT_THICKNESS,
    )


def draw_line(frame, detection1, detection2):
    cv2.line(frame, detection1.centroid, detection2.centroid, BLUE, LINE_THICKNESS)


class Detection:
    def __init__(self, box, class_id, id):

        self.box = box
        self.class_id = class_id
        self.id = id
        self.centroid = get_centroid(box)

    def get_label(self):
        return model.names[self.class_id]

    def get_distance_to(self, other):
        return math.dist(self.centroid, other.centroid)

    def draw(self, frame, is_sorted, is_alone):
        color = YELLOW if is_alone else RED
        if is_sorted and not is_alone:
            draw_box(frame, self.box, GREEN)
        else:
            draw_box(frame, self.box, color)
            draw_label(frame, self.box, self.get_label(), color)


class Cluster:
    def __init__(self):
        self.detections = []

    def add_detection(self, detection):
        self.detections.append(detection)

    def get_box(self):
        x1 = min(d.box[0] for d in self.detections)
        y1 = min(d.box[1] for d in self.detections)
        x2 = max(d.box[2] for d in self.detections)
        y2 = max(d.box[3] for d in self.detections)
        return (x1, y1, x2, y2)

    def get_centroid(self):
        return get_centroid(self.get_box())

    def is_sorted(self):
        label = self.detections[0].get_label()
        return all(d.get_label() == label for d in self.detections)

    def draw(self, frame):
        if DEBUG:
            color = GREEN if self.is_sorted() else RED
            draw_box(frame, self.get_box(), color, BOX_PADDING * 2)
            draw_label(
                frame,
                self.get_box(),
                f"{len(self.detections)} items",
                color,
                BOX_PADDING * 4,
            )

            for d1, d2 in combinations(self.detections, 2):
                draw_line(frame, d1, d2)

        is_sorted = self.is_sorted()
        has_single_detection = len(self.detections) == 1
        if is_sorted and not has_single_detection:
            draw_label(
                frame,
                self.get_box(),
                PLURAL_LABELS.get(self.detections[0].get_label()),
                GREEN,
                BOX_PADDING * 2,
            )

        for detection in self.detections:
            detection.draw(frame, is_sorted, has_single_detection)


def extract_detections(results):
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                id = int(box.id.cpu())
                detections.append(Detection((x1, y1, x2, y2), class_id, id))
    return detections


def process_frame(frame, results):
    annotated_frame = frame.copy()
    detections = extract_detections(results)

    # dynamic threshold based on resolution
    diagonal = math.sqrt(frame.shape[0] ** 2 + frame.shape[1] ** 2)
    cluster_threshold = diagonal * CLUSTER_THRESHOLD

    clusters = []

    if not detections:
        return annotated_frame

    for detection in detections:

        added_to_cluster = False

        for cluster in clusters:
            # if any detection in a cluster is close enough add it to the cluster
            if any(
                detection.get_distance_to(d) < cluster_threshold
                for d in cluster.detections
            ):
                cluster.add_detection(detection)
                added_to_cluster = True
                break

        if not added_to_cluster:
            new_cluster = Cluster()
            new_cluster.add_detection(detection)
            clusters.append(new_cluster)

    for cluster in clusters:
        cluster.draw(annotated_frame)

    return annotated_frame


def process_video(model, input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps} FPS")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # avc1 to open in vs code
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            verbose=False,
            persist=True,
            tracker="bytetrack.yaml",
        )
        annotated_frame = process_frame(frame, results)
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")


def main():
    process_video(model, input_path, output_path)


if __name__ == "__main__":
    main()
