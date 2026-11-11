"""Load saved CNN bundles and classify the currently opened image."""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utils import figure_to_image
from .classification import DEFAULT_DATA, IMAGE_SIZE, get_keras, log_progress, prepare_image


def choose_model(parent: tk.Misc) -> str:
	return filedialog.askopenfilename(
		parent=parent, title="Choose a trusted trained model passport", initialdir=str(DEFAULT_DATA),
		filetypes=[("Model passport", "passport.json")],
	)


def load_cnn(passport_path: Path):
	log_progress(f"Reading model passport: {passport_path}")
	passport = json.loads(passport_path.read_text(encoding="utf-8"))
	if not isinstance(passport, dict) or passport.get("format") != "classification_cnn_v1":
		raise ValueError("Choose a model passport saved by Train classification CNN.")
	class_names = passport.get("class_names")
	if not isinstance(class_names, list) or len(class_names) < 2 or any(not isinstance(name, str) or not name for name in class_names):
		raise ValueError("The model passport must contain at least two class names.")
	if len(set(class_names)) != len(class_names):
		raise ValueError("Model class names must not repeat.")
	preprocessing = passport.get("preprocessing", {})
	if not isinstance(preprocessing, dict) or (
		preprocessing.get("input_shape") != [IMAGE_SIZE, IMAGE_SIZE, 3]
		or preprocessing.get("color_mode") != "RGB" or preprocessing.get("dtype") != "float32"
		or preprocessing.get("normalization") != "Divide pixel values by 255.0"
	):
		raise ValueError("Unsupported model preprocessing. Expected 32 x 32 RGB float32 pixels divided by 255.")
	if passport.get("architecture") != "architecture.json" or passport.get("weights") != "model.weights.h5":
		raise ValueError("The passport must reference architecture.json and model.weights.h5 in its own folder.")
	folder = passport_path.parent
	log_progress("Reconstructing saved model architecture...")
	model = get_keras().models.model_from_json((folder / "architecture.json").read_text(encoding="utf-8"))
	log_progress(f"Architecture ready. Loading trained weights: {folder / 'model.weights.h5'}")
	model.load_weights(folder / "model.weights.h5")
	if model.input_shape != (None, IMAGE_SIZE, IMAGE_SIZE, 3) or model.output_shape != (None, len(class_names)):
		raise ValueError("Model shapes do not match the passport's preprocessing and class names.")
	log_progress("Saved model loaded and passport shapes verified.")
	return model, passport


def passport_details(passport: dict) -> dict[str, object]:
	details: dict[str, object] = {}

	def add_values(values: dict, prefix: str) -> None:
		for key, value in values.items():
			name = f"{prefix}.{key}"
			if isinstance(value, dict):
				add_values(value, name)
			else:
				details[name] = json.dumps(value) if isinstance(value, (list, tuple)) else value

	add_values(passport, "Passport")
	return details


def predict_image(image: Image.Image, model, passport: dict) -> tuple[str, dict[str, float]]:
	log_progress("Preprocessing opened image and running CNN prediction...")
	pixels = prepare_image(image)[None, ...]
	probabilities = np.array(model(pixels, training=False).numpy(), dtype=float)
	if probabilities.shape != (1, len(passport["class_names"])) or not np.isfinite(probabilities).all():
		raise ValueError("The model did not return one finite probability per class.")
	if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1, atol=1e-4):
		raise ValueError("The model output must be softmax class probabilities.")
	scores = {name: float(value) for name, value in zip(passport["class_names"], probabilities[0])}
	log_progress("CNN prediction completed.")
	return max(scores, key=lambda name: scores[name]), scores


def render_prediction(image: Image.Image, label: str, scores: dict[str, float]) -> Image.Image:
	figure = Figure(figsize=(10, 5), dpi=100, layout="constrained")
	image_axis, score_axis = figure.subplots(1, 2)
	image_axis.imshow(image.convert("RGB"))
	image_axis.set_title(f"{label} ({scores[label]:.1%})", wrap=True)
	image_axis.set_axis_off()
	names = sorted(scores, key=lambda name: scores[name])
	score_axis.barh(names, [scores[name] for name in names])
	score_axis.set(xlim=(0, 1), xlabel="Probability", title="Class probabilities")
	return figure_to_image(figure)


class CNNPredictionTool(ForensicsTool):
	tool_id = "cnn_prediction"
	title = "Predict image class"
	category = "Set 8"
	description = "Classify the opened image with a saved CNN and display its model passport."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		model_name = choose_model(parent)
		if not model_name:
			return None
		model, passport = load_cnn(Path(model_name))
		label, scores = predict_image(document.current, model, passport)
		return ToolResult(
			image=render_prediction(document.current, label, scores), message=f"Predicted {label} ({scores[label]:.1%}).",
			details={
				"Operation": "CNN image classification", "Model passport": model_name,
				"Test image": "Currently open image", "Prediction": label, "Top probability": scores[label],
				**{f"p({name})": score for name, score in scores.items()}, **passport_details(passport),
			},
		)