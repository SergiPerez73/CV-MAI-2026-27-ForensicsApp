"""Extract and save test-set CNN feature vectors with labels and source paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from .classification import DEFAULT_DATA, IMAGE_SUFFIXES, get_keras, log_progress, prepare_image
from .prediction import choose_model, load_cnn


@dataclass(frozen=True)
class FeatureVectors:
	features: np.ndarray
	labels: np.ndarray
	paths: np.ndarray
	class_names: tuple[str, ...]
	metadata: dict

	def __post_init__(self) -> None:
		if self.features.ndim != 2 or min(self.features.shape) < 1 or not np.isfinite(self.features).all():
			raise ValueError("Features must be a nonempty, finite N x D matrix.")
		count = len(self.features)
		if self.labels.shape != (count,) or not np.issubdtype(self.labels.dtype, np.integer):
			raise ValueError("Feature labels must contain one integer per vector.")
		if not self.class_names or any(not isinstance(name, str) or not name for name in self.class_names):
			raise ValueError("Feature vectors need nonempty class names.")
		if len(set(self.class_names)) != len(self.class_names) or self.labels.min() < 0 or self.labels.max() >= len(self.class_names):
			raise ValueError("Feature labels do not match distinct class names.")
		if self.paths.shape != (count,) or self.paths.dtype.kind not in "US":
			raise ValueError("Feature paths must contain one string per vector.")
		if not isinstance(self.metadata, dict):
			raise ValueError("Feature metadata must be a dictionary.")


def extract_features(test_folder: Path, model, passport: dict, batch_size: int = 32) -> FeatureVectors:
	log_progress(f"Scanning test images for feature extraction: {test_folder}")
	if batch_size < 1:
		raise ValueError("Feature extraction batch size must be positive.")
	class_names = tuple(passport["class_names"])
	unknown = [path.name for path in test_folder.iterdir() if path.is_dir() and path.name not in class_names]
	if unknown:
		raise ValueError(f"Unknown test class folders: {', '.join(unknown)}")
	paths = []
	labels = []
	for label, name in enumerate(class_names):
		folder = test_folder / name
		if not folder.is_dir():
			raise ValueError(f"Missing test class folder: {folder}")
		class_paths = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
		if not class_paths:
			raise ValueError(f"No test images found for {name}.")
		paths.extend(class_paths)
		labels.extend([label] * len(class_paths))
	layer_name = passport.get("feature_layer", "features")
	log_progress(f"Found {len(paths)} test images. Building feature extractor for layer {layer_name}...")
	try:
		layer = model.get_layer(layer_name)
	except ValueError as error:
		raise ValueError(f"Model feature layer not found: {layer_name}") from error
	extractor = get_keras().Model(inputs=model.inputs[0], outputs=layer.output)
	log_progress(f"Extracting test features in batches of {batch_size}; first batch may initialize TensorFlow kernels...")
	features = []
	for offset in range(0, len(paths), batch_size):
		images = []
		for path in paths[offset:offset + batch_size]:
			with Image.open(path) as image:
				images.append(prepare_image(image))
		vectors = np.array(extractor(np.array(images, dtype=np.float32), training=False).numpy(), dtype=np.float32)
		if vectors.ndim != 2 or len(vectors) != len(images):
			raise ValueError("The selected model layer must return one flat feature vector per image.")
		features.append(vectors)
		if offset == 0 or (offset // batch_size + 1) % 10 == 0 or offset + batch_size >= len(paths):
			log_progress(f"Features extracted: {min(offset + batch_size, len(paths))}/{len(paths)} test images.")
	return FeatureVectors(
		np.concatenate(features), np.array(labels, dtype=np.int64),
		np.array([str(path.resolve()) for path in paths], dtype=str), class_names,
		{"test_folder": str(test_folder.resolve()), "feature_layer": layer_name, "passport": passport},
	)


def save_features(path: Path, vectors: FeatureVectors) -> None:
	log_progress(f"Compressing and saving {len(vectors.features)} feature vectors to: {path}")
	np.savez_compressed(
		path, format="cnn_features_v1", features=vectors.features, labels=vectors.labels, paths=vectors.paths,
		class_names=np.array(vectors.class_names, dtype=str), metadata=json.dumps(vectors.metadata, allow_nan=False),
	)
	log_progress("Feature archive saved.")


def load_features(path: Path) -> FeatureVectors:
	log_progress(f"Loading feature archive: {path}")
	try:
		with np.load(path, allow_pickle=False) as archive:
			if str(archive["format"]) != "cnn_features_v1":
				raise ValueError("Unsupported feature archive format.")
			names = np.array(archive["class_names"])
			if names.ndim != 1 or names.dtype.kind not in "US":
				raise ValueError("Invalid feature class names.")
			vectors = FeatureVectors(
				np.array(archive["features"], dtype=np.float32), np.array(archive["labels"]), np.array(archive["paths"]),
				tuple(str(name) for name in names), json.loads(str(archive["metadata"])),
			)
			log_progress(f"Feature archive ready: {len(vectors.features)} vectors, {vectors.features.shape[1]} dimensions.")
			return vectors
	except (KeyError, TypeError, ValueError) as error:
			raise ValueError(f"Invalid CNN feature archive: {error}") from error


class CNNFeatureExtractionTool(ForensicsTool):
	tool_id = "cnn_feature_extraction"
	title = "Extract test features"
	category = "Set 8"
	description = "Extract saved-model feature vectors from test class folders and save them with labels."
	requires_image = False

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		model_name = choose_model(parent)
		if not model_name:
			return None
		folder_name = filedialog.askdirectory(parent=parent, title="Choose test folder with class subfolders", initialdir=str(DEFAULT_DATA / "test"))
		if not folder_name:
			return None
		batch_size = simpledialog.askinteger("Feature extraction", "Batch size:", parent=parent, initialvalue=32, minvalue=1)
		if batch_size is None:
			return None
		output_name = filedialog.asksaveasfilename(
			parent=parent, title="Save test feature vectors", initialdir=str(Path(model_name).parent),
			initialfile=f"test_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz",
			defaultextension=".npz", filetypes=[("CNN feature vectors", "*.npz")],
		)
		if not output_name:
			return None
		model, passport = load_cnn(Path(model_name))
		vectors = extract_features(Path(folder_name), model, passport, batch_size)
		vectors.metadata["model_passport"] = str(Path(model_name).resolve())
		save_features(Path(output_name), vectors)
		return ToolResult(
			message=f"Saved {len(vectors.features)} test feature vectors to {output_name}.",
			details={
				"Operation": "CNN test feature extraction", "Model passport": model_name, "Test folder": folder_name,
				"Feature archive": output_name, "Feature layer": vectors.metadata["feature_layer"],
				"Test images": len(vectors.features), "Feature dimensions": vectors.features.shape[1], "Batch size": batch_size,
				**{f"Images: {name}": int(np.sum(vectors.labels == label)) for label, name in enumerate(vectors.class_names)},
			},
		)