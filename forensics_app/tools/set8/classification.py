"""Train classification CNNs and save their best validation weights and model passport."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from sklearn.model_selection import train_test_split

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utils import (
	add_dialog_buttons, add_numeric_field, choose_option, create_dialog, figure_to_image, wait_for_dialog,
)


DEFAULT_DATA = Path(__file__).resolve().parents[3] / "sets" / "Set8_images"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = 32
ARCHITECTURES = {"small_cnn": "Small CNN", "vgg_cnn": "VGG-style CNN", "inception_v3": "InceptionV3"}
TRAINING_MODES = {"scratch": "From scratch", "transfer": "Transfer learning (ImageNet)"}
SELECTION_METRICS = {"val_loss": "Validation loss (minimum)", "val_accuracy": "Validation accuracy (maximum)"}


@dataclass(frozen=True)
class CNNSettings:
	epochs: int = 50
	batch_size: int = 32
	learning_rate: float = 0.01
	validation_fraction: float = 0.1
	seed: int = 42
	selection_metric: str = "val_loss"
	architecture: str = "small_cnn"
	training_mode: str = "scratch"
	samples_per_class: int = 0

	def __post_init__(self) -> None:
		if not isinstance(self.samples_per_class, int) or self.samples_per_class < 0 or self.samples_per_class == 1:
			raise ValueError("Samples per class must be 0 (all images) or an integer of at least 2.")
		if self.architecture not in ARCHITECTURES or self.training_mode not in TRAINING_MODES:
			raise ValueError("Select a supported architecture and training mode.")
		if self.training_mode == "transfer" and self.architecture != "inception_v3":
			raise ValueError("ImageNet transfer learning requires the InceptionV3 architecture.")
		if self.selection_metric not in {"val_loss", "val_accuracy"}:
			raise ValueError("Select validation loss or validation accuracy for the best model.")
		if self.epochs < 1 or self.batch_size < 1:
			raise ValueError("Epochs and batch size must be positive.")
		if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
			raise ValueError("Learning rate must be finite and positive.")
		if not np.isfinite(self.validation_fraction) or not 0 < self.validation_fraction < 1:
			raise ValueError("Validation fraction must be between 0 and 1.")
		if not 0 <= self.seed < 2 ** 32:
			raise ValueError("Random seed must be between 0 and 2**32 - 1.")


def log_progress(message: str) -> None:
	print(f"[{datetime.now().strftime('%H:%M:%S')}] [Set 8] {message}", flush=True)


def select_sample_indices(labels: np.ndarray, settings: CNNSettings) -> np.ndarray:
	if settings.samples_per_class == 0:
		return np.arange(len(labels))
	random = np.random.default_rng(settings.seed)
	selected = []
	for label in np.unique(labels):
		indices = np.flatnonzero(labels == label)
		if len(indices) > settings.samples_per_class:
			indices = random.choice(indices, settings.samples_per_class, replace=False)
		selected.extend(indices)
	return np.sort(np.array(selected, dtype=np.int64))


def get_keras():
	log_progress("Loading TensorFlow/Keras runtime...")
	try:
		import tensorflow as tf
	except ImportError as error:
		raise ValueError("CNN training needs TensorFlow. Install it with: python -m pip install 'tensorflow>=2.16,<3'") from error
	keras = tf.keras
	log_progress("TensorFlow/Keras runtime ready.")
	return keras


def prepare_image(image: Image.Image) -> np.ndarray:
	image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
	return np.array(image, dtype=np.float32) / 255.0


def load_training_images(folder: Path, settings: CNNSettings = CNNSettings()) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
	log_progress(f"Scanning training folder: {folder}")
	class_folders = sorted(path for path in folder.iterdir() if path.is_dir())
	if len(class_folders) < 2:
		raise ValueError("Choose a training folder containing at least two class subfolders.")
	images = []
	labels = []
	random = np.random.default_rng(settings.seed)
	for label, class_folder in enumerate(class_folders):
		paths = sorted(path for path in class_folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
		if len(paths) < 2:
			raise ValueError(f"At least two images are needed for class {class_folder.name}.")
		available = len(paths)
		if settings.samples_per_class and available > settings.samples_per_class:
			indices = sorted(random.choice(available, settings.samples_per_class, replace=False))
			paths = [paths[index] for index in indices]
		log_progress(f"Loading class {class_folder.name}: {len(paths)} selected / {available} available images.")
		for index, path in enumerate(paths, start=1):
			with Image.open(path) as image:
				images.append(prepare_image(image))
			labels.append(label)
			if index % 250 == 0 or index == len(paths):
				log_progress(f"Loaded {class_folder.name}: {index}/{len(paths)} images.")
	log_progress(f"Training images loaded: {len(images)} images, {len(class_folders)} classes.")
	return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64), tuple(path.name for path in class_folders)


def build_cnn(class_count: int, settings: CNNSettings = CNNSettings()):
	if class_count < 2:
		raise ValueError("At least two classes are needed for classification.")
	keras = get_keras()
	keras.utils.set_random_seed(settings.seed)
	log_progress(f"Building {ARCHITECTURES[settings.architecture]} ({TRAINING_MODES[settings.training_mode]}).")
	if settings.architecture == "inception_v3":
		transfer = settings.training_mode == "transfer"
		log_progress("Loading InceptionV3 ImageNet weights (Keras cache or first-use download)..." if transfer else "Initializing InceptionV3 with random weights...")
		base_model = keras.applications.InceptionV3(
			include_top=False, weights="imagenet" if transfer else None, input_shape=(128, 128, 3),
		)
		base_model.trainable = not transfer
		log_progress(f"InceptionV3 base ready; frozen={transfer}. Building classification head...")
		inputs = keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
		resized = keras.layers.Resizing(128, 128, interpolation="bilinear", name="resize_inception")(inputs)
		scaled = keras.layers.Rescaling(2.0, offset=-1.0, name="inception_preprocessing")(resized)
		feature_maps = base_model(scaled, training=False) if transfer else base_model(scaled)
		pooled = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(feature_maps)
		features = keras.layers.Dense(1024, activation="relu", name="features")(pooled)
		outputs = keras.layers.Dense(class_count, activation="softmax", name="predictions")(features)
		model = keras.Model(inputs, outputs, name="inception_classifier")
		optimizer = keras.optimizers.Adam(learning_rate=settings.learning_rate)
	else:
		layers = [
			keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
			keras.layers.Conv2D(
				32, (3, 3), activation="relu", kernel_initializer="he_uniform",
				padding="same" if settings.architecture == "vgg_cnn" else "valid",
			),
		]
		if settings.architecture == "vgg_cnn":
			layers.append(keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
		layers.extend([
			keras.layers.MaxPooling2D((2, 2)), keras.layers.Flatten(),
			keras.layers.Dense(
				128 if settings.architecture == "vgg_cnn" else 100,
				activation="relu", kernel_initializer="he_uniform", name="features",
			),
			keras.layers.Dense(class_count, activation="softmax", name="predictions"),
		])
		model = keras.Sequential(layers, name=settings.architecture)
		optimizer = keras.optimizers.SGD(learning_rate=settings.learning_rate)
	log_progress("Compiling model...")
	model.compile(
		loss="categorical_crossentropy", optimizer=optimizer,
		metrics=["accuracy"],
	)
	log_progress(f"Model ready: {model.count_params():,} parameters.")
	return model


def train_cnn(
	images: np.ndarray, labels: np.ndarray, class_names: tuple[str, ...], dataset: str,
	settings: CNNSettings = CNNSettings(),
):
	log_progress(f"Preparing training data: {len(images)} images.")
	if images.ndim != 4 or images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3) or len(images) == 0:
		raise ValueError("Training images must have shape (N, 32, 32, 3).")
	if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
		raise ValueError("Training pixels must be finite values in [0, 1].")
	if labels.shape != (len(images),) or not np.issubdtype(labels.dtype, np.integer):
		raise ValueError("Provide one integer class label per image.")
	if len(class_names) < 2 or len(set(class_names)) != len(class_names):
		raise ValueError("Provide at least two distinct class names.")
	counts = np.bincount(labels) if len(labels) and labels.min() >= 0 else np.array([])
	if len(counts) != len(class_names) or (counts < 2).any():
		raise ValueError("Each named class needs at least two images with labels in class-name order.")
	indices = select_sample_indices(labels, settings)
	if len(indices) < len(labels):
		images, labels = images[indices], labels[indices]
		counts = np.bincount(labels, minlength=len(class_names))
		log_progress(f"Applied sample limit: {len(labels)} images, up to {settings.samples_per_class} per class.")
	validation_size = max(len(class_names), int(np.ceil(len(labels) * settings.validation_fraction)))
	if len(labels) - validation_size < len(class_names):
		raise ValueError("The validation split leaves too few training images. Reduce the validation fraction.")
	train_indices, validation_indices = train_test_split(
		np.arange(len(labels)), test_size=validation_size, stratify=labels, random_state=settings.seed,
	)
	log_progress(f"Split ready: {len(train_indices)} training / {len(validation_indices)} validation images.")
	one_hot_labels = np.eye(len(class_names), dtype=np.float32)[labels]
	model = build_cnn(len(class_names), settings)
	mode = "min" if settings.selection_metric == "val_loss" else "max"
	with TemporaryDirectory() as folder:
		checkpoint_path = Path(folder) / "best.weights.h5"
		checkpoint = get_keras().callbacks.ModelCheckpoint(
			str(checkpoint_path), monitor=settings.selection_metric, mode=mode,
			save_best_only=True, save_weights_only=True, verbose=1,
		)
		keras = get_keras()

		class TrainingProgress(keras.callbacks.Callback):
			def on_epoch_begin(self, epoch, logs=None):
				log_progress(f"Epoch {epoch + 1}/{settings.epochs} started; running training batches...")

			def on_train_batch_end(self, batch, logs=None):
				if batch == 0 or (batch + 1) % 100 == 0:
					log_progress(f"Training batch {batch + 1}/{int(np.ceil(len(train_indices) / settings.batch_size))} completed.")

			def on_test_begin(self, logs=None):
				log_progress("Running validation batches...")

			def on_epoch_end(self, epoch, logs=None):
				log_progress(f"Epoch {epoch + 1} completed: {logs}")

		log_progress(f"Starting fit: {settings.epochs} epochs, batch size {settings.batch_size}, selecting best {settings.selection_metric}.")
		history = model.fit(
			images[train_indices], one_hot_labels[train_indices], epochs=settings.epochs, batch_size=settings.batch_size,
			validation_data=(images[validation_indices], one_hot_labels[validation_indices]), callbacks=[checkpoint, TrainingProgress()], verbose=2,
		)
		log_progress("Training finished. Restoring best-validation checkpoint...")
		model.load_weights(checkpoint_path)
		log_progress("Best-validation weights restored.")
	selection_values = np.array(history.history[settings.selection_metric], dtype=float)
	best_index = int(np.argmin(selection_values) if mode == "min" else np.argmax(selection_values))
	log_progress(f"Best epoch: {best_index + 1}; {settings.selection_metric}={selection_values[best_index]:.6f}.")
	passport = {
		"format": "classification_cnn_v1", "architecture": "architecture.json", "weights": "model.weights.h5",
		"created_at": datetime.now().isoformat(), "dataset": dataset, "sample_size": len(labels),
		"training_samples": len(train_indices), "validation_samples": len(validation_indices),
		"class_names": list(class_names),
		"samples_per_class": {name: int(count) for name, count in zip(class_names, counts)},
		"training_samples_per_class": {name: int(np.sum(labels[train_indices] == index)) for index, name in enumerate(class_names)},
		"validation_samples_per_class": {name: int(np.sum(labels[validation_indices] == index)) for index, name in enumerate(class_names)},
		"preprocessing": {
			"color_mode": "RGB", "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 3], "dtype": "float32",
			"resize": "Pillow bilinear resize to 32 x 32; aspect ratio is not preserved",
			"normalization": "Divide pixel values by 255.0", "pixel_range": [0, 1],
			"labels": "One-hot encoded in class_names order",
		},
		"settings": asdict(settings), "optimizer": "Adam" if settings.architecture == "inception_v3" else "SGD",
		"loss": "categorical_crossentropy", "model_architecture": settings.architecture,
		"training_mode": settings.training_mode, "pretrained_weights": "ImageNet" if settings.training_mode == "transfer" else None,
		"base_model_frozen": settings.training_mode == "transfer",
		"internal_preprocessing": (
			"Resize to 128 x 128 using bilinear interpolation; scale [0, 1] to [-1, 1]"
			if settings.architecture == "inception_v3" else "None"
		),
		"feature_layer": "features",
		"model_selection": {
			"metric": settings.selection_metric, "mode": mode, "best_epoch": best_index + 1,
			"best_value": float(selection_values[best_index]), "weights": "Best validation epoch; first epoch wins ties",
		},
		"training_history": {name: [float(value) for value in values] for name, values in history.history.items()},
	}
	return model, passport


def describe_model(model) -> dict[str, object]:
	details: dict[str, object] = {
		"Model input shape": str(model.input_shape), "Model output shape": str(model.output_shape),
		"Total parameters": int(model.count_params()),
		"Trainable parameters": sum(int(np.prod(weight.shape)) for weight in model.trainable_weights),
		"Non-trainable parameters": sum(int(np.prod(weight.shape)) for weight in model.non_trainable_weights),
	}
	config_keys = (
		"filters", "kernel_size", "strides", "padding", "pool_size", "units", "activation",
		"axis", "epsilon", "momentum", "rate", "height", "width", "interpolation", "scale", "offset",
	)

	def add_layers(layers, prefix: str = "") -> None:
		for index, layer in enumerate(layers, start=1):
			number = f"{prefix}{index}"
			config = layer.get_config()
			settings = {key: config[key] for key in config_keys if key in config}
			output = layer.output
			shape = [tuple(tensor.shape) for tensor in output] if isinstance(output, (list, tuple)) else tuple(output.shape)
			value = (
				f"{type(layer).__name__}; output={shape}; parameters={layer.count_params()}; trainable={layer.trainable}"
			)
			if settings:
				value += f"; {json.dumps(settings)}"
			details[f"Layer {number}: {layer.name}"] = value
			if hasattr(layer, "layers"):
				add_layers(layer.layers, prefix=f"{number}.")

	add_layers(model.layers)
	return details


def save_cnn(folder: Path, model, passport: dict) -> None:
	log_progress(f"Saving model bundle to: {folder}")
	folder.mkdir(parents=True, exist_ok=True)
	log_progress("Writing model.weights.h5...")
	model.save_weights(folder / "model.weights.h5")
	log_progress("Writing architecture.json and passport.json...")
	(folder / "architecture.json").write_text(model.to_json(), encoding="utf-8")
	(folder / "passport.json").write_text(json.dumps(passport, indent=2, allow_nan=False), encoding="utf-8")
	summary = []
	model.summary(print_fn=summary.append, expand_nested=True, show_trainable=True, line_length=140)
	(folder / "model_summary.txt").write_text("\n".join(summary), encoding="utf-8")
	log_progress("Model bundle saved.")


def render_training_history(history: dict) -> Image.Image:
	figure = Figure(figsize=(9, 4), dpi=100, layout="constrained")
	axes = figure.subplots(1, 2)
	for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Cross entropy loss", "Classification accuracy")):
		for key, label in ((metric, "Training"), (f"val_{metric}", "Validation")):
			values = history[key]
			axis.plot(range(1, len(values) + 1), values, label=label)
		axis.set_title(title)
		axis.set_xlabel("Epoch")
		axis.legend()
	return figure_to_image(figure)


def choose_cnn_settings(parent: tk.Misc, *, title: str) -> CNNSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: CNNSettings | None = None
	defaults = CNNSettings()
	fields = (
		("Epochs:", str(defaults.epochs)), ("Batch size:", str(defaults.batch_size)),
		("Learning rate:", str(defaults.learning_rate)), ("Validation fraction:", str(defaults.validation_fraction)),
		("Samples per class (0 = all):", str(defaults.samples_per_class)),
	)
	values = [tk.StringVar(master=dialog, value=default) for _, default in fields]
	architecture = tk.StringVar(master=dialog, value=ARCHITECTURES[defaults.architecture])
	training_mode = tk.StringVar(master=dialog, value=TRAINING_MODES[defaults.training_mode])
	selection_metric = tk.StringVar(master=dialog, value=SELECTION_METRICS[defaults.selection_metric])
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			selected = CNNSettings(
				epochs=int(values[0].get()), batch_size=int(values[1].get()),
				learning_rate=float(values[2].get()), validation_fraction=float(values[3].get()),
				samples_per_class=int(values[4].get()),
				architecture=next(key for key, label in ARCHITECTURES.items() if label == architecture.get()),
				training_mode=next(key for key, label in TRAINING_MODES.items() if label == training_mode.get()),
				selection_metric=next(key for key, label in SELECTION_METRICS.items() if label == selection_metric.get()),
			)
		except ValueError as error:
			error_text.set(str(error))
			return
		dialog.destroy()

	controls = []
	for label, value, choices in (
		("Architecture:", architecture, ARCHITECTURES), ("Training mode:", training_mode, TRAINING_MODES),
		("Save best model by:", selection_metric, SELECTION_METRICS),
	):
		row = ttk.Frame(dialog)
		row.pack(fill="x", padx=16, pady=(12, 0))
		ttk.Label(row, text=label).pack(side="left", padx=(0, 12))
		control = ttk.Combobox(row, textvariable=value, values=list(choices.values()), state="readonly", width=30)
		control.pack(side="right")
		controls.append(control)

	def update_architecture(_event=None) -> None:
		inception = architecture.get() == ARCHITECTURES["inception_v3"]
		controls[1].configure(values=list(TRAINING_MODES.values()) if inception else [TRAINING_MODES["scratch"]])
		if not inception:
			training_mode.set(TRAINING_MODES["scratch"])
		values[2].set("0.001" if inception else str(defaults.learning_rate))

	controls[0].bind("<<ComboboxSelected>>", update_architecture)
	update_architecture()
	for (label, _), value in zip(fields, values):
		add_numeric_field(dialog, label, value)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


class CNNTrainingTool(ForensicsTool):
	tool_id = "cnn_training"
	title = "Train classification CNN"
	category = "Set 8"
	description = "Train a small CNN and save its weights, architecture, preprocessing, and training history."
	requires_image = False

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		source = choose_option(
			parent, title=self.title, prompt="Training dataset:",
			options={"folder": "Class folders", "cifar10": "CIFAR-10 (download)"},
		)
		if source is None:
			return None
		folder_name = ""
		if source == "folder":
			folder_name = filedialog.askdirectory(
				parent=parent, title="Choose training folder with class subfolders", initialdir=str(DEFAULT_DATA / "train"),
			)
			if not folder_name:
				return None
		settings = choose_cnn_settings(parent, title=self.title)
		if settings is None:
			return None
		get_keras()
		if source == "cifar10":
			log_progress("Loading CIFAR-10 from Keras cache or downloading it on first use...")
			(images, labels), _ = get_keras().datasets.cifar10.load_data()
			labels = np.array(labels, dtype=np.int64).reshape(-1)
			log_progress(f"CIFAR-10 loaded: {len(images)} training images. Selecting samples and normalizing...")
			indices = select_sample_indices(labels, settings)
			images = np.array(images[indices], dtype=np.float32) / 255.0
			labels = labels[indices]
			log_progress(f"CIFAR-10 prepared: {len(images)} selected images.")
			class_names = CLASSES
			dataset = "CIFAR-10 official training split (test split excluded)"
		else:
			images, labels, class_names = load_training_images(Path(folder_name), settings)
			dataset = str(Path(folder_name).resolve())
		model, passport = train_cnn(images, labels, class_names, dataset, settings)
		output_folder = DEFAULT_DATA / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
		save_cnn(output_folder, model, passport)
		history = passport["training_history"]
		best_index = passport["model_selection"]["best_epoch"] - 1
		log_progress("Rendering training curves and preparing Results...")
		return ToolResult(
			image=render_training_history(history), message=f"CNN trained. Saved weights and model passport to {output_folder}.",
			details={
				"Operation": "Classification CNN training", "Dataset": dataset, "Class names": ", ".join(class_names),
				"Training images": passport["training_samples"], "Validation images": passport["validation_samples"],
				"Samples per class limit": settings.samples_per_class or "All",
				"Epochs": len(history["loss"]), "Training accuracy": history["accuracy"][best_index],
				"Validation accuracy": history["val_accuracy"][best_index], "Validation loss": history["val_loss"][best_index],
				"Best epoch": best_index + 1, "Selection metric": settings.selection_metric,
				"Model architecture": ARCHITECTURES[settings.architecture], "Training mode": TRAINING_MODES[settings.training_mode],
				"Weights": str(output_folder / "model.weights.h5"),
				"Architecture": str(output_folder / "architecture.json"), "Model passport": str(output_folder / "passport.json"),
				"Model summary": str(output_folder / "model_summary.txt"),
				**describe_model(model),
			},
		)