"""Detect faces with a Haar/AdaBoost cascade or YuNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utils import add_dialog_buttons, add_numeric_field, choose_option, create_dialog, wait_for_dialog


METHODS = {"haar": "Haar / AdaBoost", "yunet": "YuNet"}
DEFAULT_MODEL = (
	Path(__file__).resolve().parents[3]
	/ "sets" / "Set6_images" / "face_detection_yunet_2023mar.onnx"
)


@dataclass(frozen=True)
class FaceDetectionSettings:
	method: str
	# Haar scans an image pyramid: 1.1 uses scale steps of about 10%.
	scale_factor: float = 1.1
	# More supporting, overlapping detections are required at higher values.
	min_neighbors: int = 5
	# Minimum bounding-box width and height in pixels (Haar only).
	min_size: int = 30
	# YuNet discards candidates below this confidence score.
	score_threshold: float = 0.6
	# NMS removes lower-scoring boxes whose overlap (IoU) exceeds this value.
	nms_threshold: float = 0.3
	model_path: str = str(DEFAULT_MODEL)


def choose_face_detection(parent: tk.Misc, *, title: str) -> FaceDetectionSettings | None:
	method = choose_option(parent, title=title, prompt="Detection method:", options=METHODS)
	if method is None:
		return None
	dialog = create_dialog(parent, title=f"{title} - {METHODS[method]}")
	selected: FaceDetectionSettings | None = None
	scale_value = tk.StringVar(master=dialog, value="1.1")
	neighbors_value = tk.StringVar(master=dialog, value="5")
	size_value = tk.StringVar(master=dialog, value="30")
	score_value = tk.StringVar(master=dialog, value="0.6")
	nms_value = tk.StringVar(master=dialog, value="0.3")
	model_value = tk.StringVar(master=dialog, value=str(DEFAULT_MODEL))
	error_text = tk.StringVar(master=dialog)

	def choose_model() -> None:
		filename = filedialog.askopenfilename(
			parent=dialog, title="Choose YuNet model", filetypes=[("ONNX model", "*.onnx")]
		)
		if filename:
			model_value.set(filename)

	def apply_settings() -> None:
		nonlocal selected
		try:
			# Tk fields contain text; convert and validate before running OpenCV.
			settings = FaceDetectionSettings(
				method=method,
				scale_factor=float(scale_value.get()),
				min_neighbors=int(neighbors_value.get()),
				min_size=int(size_value.get()),
				score_threshold=float(score_value.get()),
				nms_threshold=float(nms_value.get()),
				model_path=model_value.get(),
			)
			validate_settings(settings)
		except ValueError as error:
			error_text.set(str(error))
			return
		selected = settings
		dialog.destroy()

	if method == "haar":
		add_numeric_field(dialog, "Scale factor (> 1):", scale_value)
		add_numeric_field(dialog, "Minimum neighbors:", neighbors_value)
		add_numeric_field(dialog, "Minimum face size (px):", size_value)
	else:
		add_numeric_field(dialog, "Confidence threshold (0-1):", score_value)
		add_numeric_field(dialog, "NMS threshold (0-1):", nms_value)
		ttk.Label(dialog, text="YuNet model:").pack(anchor="w", padx=16, pady=(12, 0))
		ttk.Entry(dialog, textvariable=model_value, width=60).pack(fill="x", padx=16)
		ttk.Button(dialog, text="Browse...", command=choose_model).pack(anchor="e", padx=16, pady=6)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


def validate_settings(settings: FaceDetectionSettings) -> None:
	if settings.method == "haar":
		if not np.isfinite(settings.scale_factor) or settings.scale_factor <= 1:
			raise ValueError("Scale factor must be a finite number greater than 1.")
		if settings.min_neighbors < 0 or settings.min_size < 1:
			raise ValueError("Neighbors must be nonnegative and face size must be positive.")
	elif settings.method == "yunet":
		if not 0 < settings.score_threshold <= 1 or not 0 < settings.nms_threshold <= 1:
			raise ValueError("Confidence and NMS thresholds must be greater than 0 and at most 1.")
		if not Path(settings.model_path).is_file():
			raise ValueError("Choose an existing YuNet ONNX model.")
	else:
		raise ValueError("Unknown face detection method.")


def detect_faces(image: Image.Image, settings: FaceDetectionSettings) -> tuple[Image.Image, int]:
	"""Run pretrained detection, returning an annotated copy and the face count."""
	validate_settings(settings)
	# PIL uses RGB. np.array makes a writable copy, so drawing leaves the source unchanged.
	rgb_image = np.array(image.convert("RGB"))
	if settings.method == "haar":
		# OpenCV ships this trained cascade; we do not train AdaBoost on each image.
		# Its stages combine Haar intensity-contrast features selected by AdaBoost,
		# using integral images internally to evaluate rectangular features quickly.
		cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
		classifier = cv2.CascadeClassifier(str(cascade_path))
		if classifier.empty():
			raise ValueError("Could not load OpenCV's pretrained Haar/AdaBoost face cascade.")
		# Haar needs intensity, not color. The cascade rejects unlikely windows early,
		# scans at multiple scales, then groups nearby candidate boxes.
		gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
		faces = classifier.detectMultiScale(
			gray_image,
			scaleFactor=settings.scale_factor,
			minNeighbors=settings.min_neighbors,
			minSize=(settings.min_size, settings.min_size),
		)
	else:
		# YuNet is a pretrained neural network. OpenCV expects its input in BGR order;
		# keep rgb_image separate so the final PIL image still has the correct colors.
		bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
		height, width = bgr_image.shape[:2]
		# The ONNX file contains the network and learned weights. Input size is
		# (width, height), and 5000 limits candidates considered before NMS.
		detector = cv2.FaceDetectorYN.create(
			settings.model_path, "", (width, height), settings.score_threshold, settings.nms_threshold, 5000
		)
		# Ignore the status value; faces contains the actual detection rows.
		_, faces = detector.detect(bgr_image)
	if faces is None:
		return Image.fromarray(rgb_image), 0
	# Haar returns an empty collection when nothing is detected; that also skips this loop.
	for face in faces:
		# Both methods start each row with (left, top, width, height).
		# rectangle() instead takes the top-left and bottom-right corners.
		x_coord, y_coord, width, height = (int(value) for value in face[:4])
		cv2.rectangle(rgb_image, (x_coord, y_coord), (x_coord + width, y_coord + height), (0, 255, 0), 2)
		if settings.method == "yunet":
			# YuNet rows have 15 values: 4 box coordinates, 10 landmark coordinates,
			# and a confidence score at index 14. Haar does not return these extras.
			cv2.putText(
				rgb_image, f"Face: {float(face[14]):.2f}", (x_coord, max(20, y_coord - 8)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
			)
			# Turn the 10 coordinates into five (x, y) pairs: two eyes, the nose,
			# and two mouth corners. Drawing colors use RGB here, like our array.
			for landmark in face[4:14].reshape(5, 2):
				cv2.circle(rgb_image, (int(landmark[0]), int(landmark[1])), 3, (0, 100, 255), -1)
	return Image.fromarray(rgb_image), len(faces)


class FaceDetectionTool(ForensicsTool):
	tool_id = "face_detection"
	title = "Face detection"
	category = "Set 6"
	description = "Detect faces using Haar/AdaBoost or YuNet, with YuNet facial landmarks."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		settings = choose_face_detection(parent, title=self.title)
		if settings is None:
			return None
		# Detection is independent of Tk; ToolResult hands the image and measurements
		# back to the existing UI, just like the other tools.
		output_image, face_count = detect_faces(document.current, settings)
		details: dict[str, str | int | float] = {
			"Operation": "Face detection",
			"Method": METHODS[settings.method],
			"Faces detected": face_count,
			"Output mode": output_image.mode,
		}
		if settings.method == "haar":
			details.update({
				"Scale factor": settings.scale_factor,
				"Minimum neighbors": settings.min_neighbors,
				"Minimum face size": settings.min_size,
			})
		else:
			details.update({
				"Confidence threshold": settings.score_threshold,
				"NMS threshold": settings.nms_threshold,
				"Model": settings.model_path,
			})
		return ToolResult(
			image=output_image,
			message=f"Detected {face_count} face(s) using {METHODS[settings.method]}.",
			details=details,
		)