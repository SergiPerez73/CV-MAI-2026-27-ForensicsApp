"""Detect objects in the current image with pretrained YOLO11n weights."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import tkinter as tk

from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult


MODEL_WEIGHTS = Path(__file__).resolve().parents[3] / "sets" / "set9_images" / "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.25


@lru_cache(maxsize=1)
def load_detector():
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError('Install YOLO support with: python -m pip install -e ".[yolo]"') from error

    if not MODEL_WEIGHTS.is_file():
        raise FileNotFoundError(f"Place the official YOLO weights at: {MODEL_WEIGHTS}")

    try:
        return YOLO(str(MODEL_WEIGHTS))
    except Exception as error:
        raise RuntimeError(f"Could not load {MODEL_WEIGHTS}: {error}") from error


class ObjectDetectionTool(ForensicsTool):
    tool_id = "yolo_object_detection"
    title = "Detect objects (YOLO)"
    category = "Set 9"
    description = "Detect common objects in the current image with pretrained YOLO11n."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None
        source = document.current.convert("RGB")
        prediction = load_detector().predict(source, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        boxes = prediction.boxes
        count = len(boxes) if boxes is not None else 0
        details = {
            "Model": MODEL_WEIGHTS.name,
            "Confidence threshold": CONFIDENCE_THRESHOLD,
            "Detections": count,
        }
        if not count:
            return ToolResult(message="No objects detected.", details=details)

        counts = Counter(prediction.names[int(class_id)] for class_id in boxes.cls.tolist())
        details["By class"] = ", ".join(f"{name}: {total}" for name, total in sorted(counts.items()))
        annotated_bgr = prediction.plot()
        return ToolResult(
            message=f"YOLO detections: {count}.",
            image=Image.fromarray(annotated_bgr[..., ::-1]),
            details=details,
        )