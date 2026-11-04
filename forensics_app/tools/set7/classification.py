"""Train, save, and run a visual-word SVM classifier."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import joblib
import numpy as np
from PIL import Image
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utils import choose_option
from .codebook import CLASSES, DEFAULT_DATA, IMAGE_SUFFIXES, CodebookModel, extract_phow, load_codebook
from .histogram import choose_codebook, render_histogram


@dataclass(frozen=True)
class VisualWordClassifier:
	codebook: CodebookModel
	classifier: Pipeline
	training_images: int


def train_classifier(training_folder: Path, codebook: CodebookModel) -> VisualWordClassifier:
	histograms = []
	labels = []
	for label in CLASSES:
		class_folder = training_folder / label
		if not class_folder.is_dir():
			raise ValueError(f"Missing training class folder: {class_folder}")
		paths = sorted(path for path in class_folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
		if len(paths) < 2:
			raise ValueError(f"At least two training images are needed for {label} to fit a probability-enabled SVM.")
		for path in paths:
			with Image.open(path) as image:
				histogram = extract_phow(image, codebook.centers, codebook.settings)
			if not histogram.any():
				raise ValueError(f"No SIFT descriptors found in training image: {path}")
			histograms.append(histogram)
			labels.append(label)
	classifier = make_pipeline(
		Normalizer(norm="l1"), AdditiveChi2Sampler(sample_steps=2),
		SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42),
	)
	classifier.fit(np.vstack(histograms), np.array(labels))
	return VisualWordClassifier(codebook, classifier, len(labels))


def save_classifier(path: Path, model: VisualWordClassifier) -> None:
	joblib.dump({"format": "visual_word_svm_v1", "model": model}, path, compress=3)


def load_classifier(path: Path) -> VisualWordClassifier:
	"""Only load classifier files created by you or another trusted source."""
	bundle = joblib.load(path)
	if not isinstance(bundle, dict) or bundle.get("format") != "visual_word_svm_v1":
		raise ValueError("Not a supported visual-word classifier.")
	model = bundle.get("model")
	if not isinstance(model, VisualWordClassifier) or not isinstance(model.codebook, CodebookModel):
		raise ValueError("The classifier is missing its visual-word model.")
	if not isinstance(model.classifier, Pipeline):
		raise ValueError("The classifier is missing its fitted SVM pipeline.")
	model.codebook.settings.__post_init__()
	model.codebook.__post_init__()
	dimensions = sum(model.codebook.settings.pyramid_levels) * model.codebook.settings.words
	if getattr(model.classifier, "n_features_in_", None) != dimensions or set(getattr(model.classifier, "classes_", ())) != set(CLASSES):
		raise ValueError("The classifier does not match its codebook or the three supported classes.")
	return model


def predict_image(image: Image.Image, model: VisualWordClassifier) -> tuple[str, float, dict[str, float], np.ndarray]:
	histogram = extract_phow(image, model.codebook.centers, model.codebook.settings)
	if not histogram.any():
		raise ValueError("No SIFT descriptors found. Use a larger image or rebuild the codebook with a smaller SIFT size.")
	probabilities = model.classifier.predict_proba(histogram.reshape(1, -1))[0]
	scores = {str(label): float(probability) for label, probability in zip(model.classifier.classes_, probabilities)}
	label = max(scores, key=lambda name: scores[name])
	return label, scores[label], scores, histogram


class VisualWordPredictionTool(ForensicsTool):
	tool_id = "visual_word_prediction"
	title = "Predict class"
	category = "Set 7"
	description = "Predict the opened image using a saved SVM, or train and save one from a visual-word model."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		mode = choose_option(
			parent, title=self.title, prompt="Classifier:",
			options={"load": "Load saved classifier", "train": "Train and save classifier"},
		)
		if mode is None:
			return None
		if mode == "load":
			model_name = filedialog.askopenfilename(
				parent=parent, title="Choose trusted classifier", initialdir=str(DEFAULT_DATA),
				filetypes=[("Visual-word classifier", "*.joblib")],
			)
			if not model_name:
				return None
			model = load_classifier(Path(model_name))
		else:
			codebook_name = choose_codebook(parent)
			if not codebook_name:
				return None
			codebook = load_codebook(Path(codebook_name))
			training_folder = filedialog.askdirectory(
				parent=parent, title="Choose training image folder", initialdir=str(DEFAULT_DATA / "train"),
			)
			if not training_folder:
				return None
			model_name = filedialog.asksaveasfilename(
				parent=parent, title="Save trained classifier", initialdir=str(Path(codebook_name).parent),
				initialfile=f"classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
				defaultextension=".joblib", filetypes=[("Visual-word classifier", "*.joblib")],
			)
			if not model_name:
				return None
			model = train_classifier(Path(training_folder), codebook)
			save_classifier(Path(model_name), model)
		label, probability, scores, histogram = predict_image(document.current, model)
		output_image = render_histogram(
			document.current, histogram, model.codebook.settings, prediction=f"{label}\np({label}) = {probability:.3f}",
		)
		details: dict[str, object] = {
			"Operation": "Visual word classification", "Classifier": model_name,
			"Test image": "Currently open image", "Prediction": label, "Top probability": probability,
			"Training images": model.training_images, "Visual words (K)": model.codebook.settings.words,
			"PHOW dimensions": len(histogram), "Normalization": "L1",
			"Algorithm": "SVM with additive chi-squared feature map and probability calibration",
		}
		for name, score in scores.items():
			details[f"p({name})"] = score
		message = f"Predicted {label} ({probability:.1%})."
		if mode == "train":
			message += f" Saved classifier to {model_name}."
		return ToolResult(image=output_image, message=message, details=details)