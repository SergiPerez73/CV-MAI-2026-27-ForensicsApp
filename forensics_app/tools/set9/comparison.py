"""Compare a blood-cell checkpoint with labels on a held-out test image."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from filecmp import cmp
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageDraw

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from .dataset import CLASS_NAMES
from .object_detection import CONFIDENCE_THRESHOLD
from .training import ASSETS


COLORS = ("#c2410c", "#16803c", "#165fce")


@dataclass(frozen=True)
class CellBox:
    class_id: int
    bounds: tuple[float, float, float, float]
    confidence: float | None = None


def read_labels(path: Path, size: tuple[int, int]) -> list[CellBox]:
    width, height = size
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            class_text, *coordinates = line.split()
            class_id = int(class_text)
            center_x, center_y, box_width, box_height = map(float, coordinates)
            if not (0 <= class_id < len(CLASS_NAMES)
                    and 0 <= center_x <= 1 and 0 <= center_y <= 1
                    and 0 < box_width <= 1 and 0 < box_height <= 1):
                raise ValueError
        except ValueError as error:
            raise ValueError(f"Invalid YOLO annotation in {path}, line {line_number}.") from error
        boxes.append(CellBox(class_id, (
            (center_x - box_width / 2) * width, (center_y - box_height / 2) * height,
            (center_x + box_width / 2) * width, (center_y + box_height / 2) * height,
        )))
    return boxes


def predict_cells(checkpoint: Path, image: Image.Image) -> list[CellBox]:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError('Install YOLO support with: python -m pip install -e ".[yolo]"') from error

    model = YOLO(str(checkpoint))
    if model.names != dict(enumerate(CLASS_NAMES)):
        raise ValueError("Choose a Set 9.2 checkpoint with classes PC, RBC, and WBC.")
    boxes = model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0].boxes
    if boxes is None:
        return []
    return [CellBox(int(class_id), (left, top, right, bottom), confidence)
            for class_id, confidence, (left, top, right, bottom) in
            zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())]


def render_comparison(image: Image.Image, labels: list[CellBox], predictions: list[CellBox]) -> Image.Image:
    width, height = image.size
    header, gap = 26, 8
    output = Image.new("RGB", (2 * width + gap, height + header), "#172026")
    for offset, title, boxes in ((0, "Ground truth", labels), (width + gap, "Predictions", predictions)):
        panel = image.copy()
        draw = ImageDraw.Draw(panel)
        for box in boxes:
            left, top, right, bottom = box.bounds
            color = COLORS[box.class_id]
            caption = CLASS_NAMES[box.class_id]
            if box.confidence is not None:
                caption += f" {box.confidence:.2f}"
            draw.rectangle((left, top, right, bottom), outline=color, width=2)
            label_left = max(0, min(left, width - draw.textlength(caption) - 4))
            label_top = max(0, top - 15)
            draw.rectangle((label_left, label_top, label_left + draw.textlength(caption) + 4, label_top + 15), fill=color)
            draw.text((label_left + 2, label_top + 1), caption, fill="white")
        output.paste(panel, (offset, header))
        ImageDraw.Draw(output).text((offset + 8, 6), title, fill="white")
    return output


def compare_test_image(run: Path, image_path: Path, *, opened_image: Image.Image | None = None) -> ToolResult:
    test = run / "dataset" / "test"
    image_path = image_path.resolve()
    test_image = (test / "images" / image_path.name).resolve()
    if not test_image.is_file() or (image_path != test_image
                                    and (not image_path.is_file() or not cmp(image_path, test_image, shallow=False))):
        raise ValueError(f"Open an image from {test / 'images'} or an identical copy.")
    checkpoint = run / "training" / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Set 9.2 checkpoint not found: {checkpoint}")
    label_path = test / "labels" / image_path.with_suffix(".txt").name
    if not label_path.is_file():
        raise FileNotFoundError(f"Ground-truth label file not found: {label_path}")

    if opened_image is None:
        with Image.open(test_image) as source:
            image = source.convert("RGB")
    else:
        image = opened_image.convert("RGB")
    labels = read_labels(label_path, image.size)
    predictions = predict_cells(checkpoint, image)
    details = {"Test image": image_path.name, "Ground-truth boxes": len(labels),
               "Predicted boxes": len(predictions), "Confidence threshold": CONFIDENCE_THRESHOLD}
    actual_counts = Counter(box.class_id for box in labels)
    predicted_counts = Counter(box.class_id for box in predictions)
    for class_id, name in enumerate(CLASS_NAMES):
        details[name] = f"{actual_counts[class_id]} ground truth / {predicted_counts[class_id]} predicted"
    return ToolResult(message=f"Compared blood-cell detections on {image_path.name}.",
                      image=render_comparison(image, labels, predictions), details=details)


class BloodCellComparisonTool(ForensicsTool):
    tool_id = "yolo_blood_cell_comparison"
    title = "Compare blood-cell detections (YOLO)"
    category = "Set 9"
    description = "Compare a trained detector's predictions with ground truth on a held-out image."
    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        if document.path is None or document.current is None:
            raise ValueError("Open a held-out test image before running the comparison.")
        if document.is_modified and document.current != document.original:
            raise ValueError("Reset or reopen the test image before comparing it with ground truth.")
        folder = filedialog.askdirectory(parent=parent, title="Choose a Set 9.2 training run",
                                         initialdir=ASSETS, mustexist=True)
        if not folder:
            return None
        return compare_test_image(Path(folder), document.path, opened_image=document.current)