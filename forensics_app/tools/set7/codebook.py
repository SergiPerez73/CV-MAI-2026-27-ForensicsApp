"""Build a visual codebook and spatial-pyramid histograms from dense SIFT."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isqrt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utils import add_dialog_buttons, add_numeric_field, create_dialog, figure_to_image, wait_for_dialog


DEFAULT_DATA = Path(__file__).resolve().parents[3] / "sets" / "Set7_images"
CLASSES = ("butterfly", "dalmatian", "kangaroo")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
WORDS = 10
GRID_STEP = 8
SIFT_SIZE = 8
PYRAMID_LEVELS = (1, 4, 16)
MAX_IMAGE_SIZE = 256


@dataclass(frozen=True)
class CodebookSettings:
	words: int = WORDS
	step: int = GRID_STEP
	sift_size: int = SIFT_SIZE
	pyramid_levels: tuple[int, ...] = PYRAMID_LEVELS
	max_image_size: int = MAX_IMAGE_SIZE
	n_octaves: int = 3
	n_scales: int = 3
	sigma_min: float = 1.6

	def __post_init__(self) -> None:
		if min(self.words, self.step, self.sift_size, self.max_image_size, self.n_octaves, self.n_scales) < 1:
			raise ValueError("K, grid step, SIFT size, image size, octaves, and scales must be positive.")
		if not self.pyramid_levels or any(level < 1 or isqrt(level) ** 2 != level for level in self.pyramid_levels):
			raise ValueError("Pyramid levels must be positive square region counts, such as 1, 4, 16.")
		if len(set(self.pyramid_levels)) != len(self.pyramid_levels):
			raise ValueError("Pyramid levels must not repeat.")
		if not np.isfinite(self.sigma_min) or self.sigma_min <= 0:
			raise ValueError("SIFT minimum sigma must be finite and positive.")


@dataclass(frozen=True)
class CodebookModel:
	centers: np.ndarray
	settings: CodebookSettings

	def __post_init__(self) -> None:
		if self.centers.shape != (self.settings.words, 128) or not np.isfinite(self.centers).all():
			raise ValueError("The codebook must contain K finite 128-value centers.")


def load_codebook(path: Path) -> CodebookModel:
	try:
		with np.load(path, allow_pickle=False) as archive:
			if str(archive["descriptor_method"]) != "spatial_pyramid_dense_sift_v3":
				raise ValueError("Unsupported descriptor method.")
			centers = np.array(archive["centers"], dtype=np.float32)
			settings = CodebookSettings(
				words=len(centers), step=int(archive["step"]), sift_size=int(archive["sift_size"]),
				pyramid_levels=tuple(int(level) for level in archive["pyramid_levels"]),
				max_image_size=int(archive["max_image_size"]), n_octaves=int(archive["n_octaves"]),
				n_scales=int(archive["n_scales"]), sigma_min=float(archive["sigma_min"]),
			)
		return CodebookModel(centers, settings)
	except (KeyError, ValueError, TypeError) as error:
		raise ValueError("Invalid or outdated codebook. Rebuild it with Codebook construction.") from error


def choose_codebook_settings(parent: tk.Misc, *, title: str) -> CodebookSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: CodebookSettings | None = None
	defaults = CodebookSettings()
	fields = (
		("Visual words (K):", str(defaults.words)),
		("Pyramid levels (region counts):", ", ".join(str(level) for level in defaults.pyramid_levels)),
		("Grid step (px):", str(defaults.step)),
		("SIFT size (bin width, px):", str(defaults.sift_size)),
		("SIFT octaves:", str(defaults.n_octaves)),
		("SIFT scales per octave:", str(defaults.n_scales)),
		("SIFT minimum sigma:", str(defaults.sigma_min)),
		("Maximum image size (px):", str(defaults.max_image_size)),
	)
	values = [tk.StringVar(master=dialog, value=default) for _, default in fields]
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			settings = CodebookSettings(
				words=int(values[0].get()),
				pyramid_levels=tuple(int(level.strip()) for level in values[1].get().split(",")),
				step=int(values[2].get()), sift_size=int(values[3].get()),
				n_octaves=int(values[4].get()), n_scales=int(values[5].get()),
				sigma_min=float(values[6].get()), max_image_size=int(values[7].get()),
			)
		except ValueError as error:
			error_text.set(str(error))
			return
		selected = settings
		dialog.destroy()

	for index, ((label, _), value) in enumerate(zip(fields, values)):
		if index == 1:
			row = ttk.Frame(dialog)
			row.pack(fill="x", padx=16, pady=4)
			ttk.Label(row, text=label).pack(side="left", padx=(0, 12))
			ttk.Entry(row, textvariable=value, width=18).pack(side="right")
		else:
			add_numeric_field(dialog, label, value)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


def prepare_image(image: Image.Image, max_image_size: int) -> np.ndarray:
	gray_image = image.convert("L")
	gray_image.thumbnail((max_image_size, max_image_size), Image.Resampling.BILINEAR)
	return np.array(gray_image, dtype=np.uint8)


def split_pyramid(image: np.ndarray, pyramid_levels: tuple[int, ...]) -> list[tuple[np.ndarray, tuple[int, int]]]:
	"""Return crops and their offsets, ordered by level, row, then column."""
	height, width = image.shape
	regions = []
	for level in pyramid_levels:
		grid_size = isqrt(level)
		if grid_size < 1 or grid_size ** 2 != level:
			raise ValueError("Each pyramid level must be a positive square region count.")
		if min(height, width) < grid_size:
			raise ValueError("The image is too small for the selected pyramid levels.")
		for row in range(grid_size):
			for column in range(grid_size):
				top, bottom = row * height // grid_size, (row + 1) * height // grid_size
				left, right = column * width // grid_size, (column + 1) * width // grid_size
				regions.append((image[top:bottom, left:right].copy(), (left, top)))
	return regions


def extract_sift(image: np.ndarray, settings: CodebookSettings) -> tuple[list, np.ndarray]:
	"""Compute upright SIFT on a dense grid in each Gaussian octave and scale."""
	height, width = image.shape
	keypoints = []
	for octave in range(settings.n_octaves):
		for scale in range(settings.n_scales):
			factor = 2 ** (octave + scale / settings.n_scales)
			bin_width = settings.sift_size * factor
			margin = int(np.ceil(1.5 * bin_width))
			step = max(1, int(round(settings.step * 2 ** octave)))
			# OpenCV's diameter is twice the scale, with a spatial bin three times the scale.
			keypoint_size = 2 * bin_width / 3
			for row in range(margin, height - margin, step):
				for column in range(margin, width - margin, step):
					keypoints.append(cv2.KeyPoint(
						float(column), float(row), keypoint_size, 0, 0, octave | (scale << 8),
					))
	if not keypoints:
		return [], np.empty((0, 128), dtype=np.float32)
	keypoints, descriptors = cv2.SIFT_create(
		nOctaveLayers=settings.n_scales, sigma=settings.sigma_min,
	).compute(image, keypoints)
	if descriptors is None:
		return [], np.empty((0, 128), dtype=np.float32)
	return list(keypoints), np.array(descriptors, dtype=np.float32)


def extract_pyramid_sift(
	image: Image.Image, settings: CodebookSettings,
) -> tuple[np.ndarray, list, list[np.ndarray]]:
	gray = prepare_image(image, settings.max_image_size)
	keypoints = []
	region_descriptors = []
	for region_index, (region, (left, top)) in enumerate(split_pyramid(gray, settings.pyramid_levels)):
		region_keypoints, descriptors = extract_sift(region, settings)
		for keypoint in region_keypoints:
			keypoint.pt = (keypoint.pt[0] + left, keypoint.pt[1] + top)
			keypoint.class_id = region_index
		keypoints.extend(region_keypoints)
		region_descriptors.append(descriptors)
	return gray, keypoints, region_descriptors


def assign_features_to_words(
	region_descriptors: list[np.ndarray], centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return region histograms and nearest-centroid labels in descriptor order."""
	words = len(centers)
	histograms = []
	region_assignments = []
	for descriptors in region_descriptors:
		if len(descriptors):
			assignments, _ = pairwise_distances_argmin_min(descriptors, centers)
			histogram = np.bincount(assignments, minlength=words)
		else:
			assignments = np.empty(0, dtype=int)
			histogram = np.zeros(words, dtype=int)
		region_assignments.append(assignments)
		histograms.append(histogram)
	return np.concatenate(histograms).astype(np.float32), np.concatenate(region_assignments)


def compute_histograms(region_descriptors: list[np.ndarray], centers: np.ndarray) -> np.ndarray:
	"""Concatenate one K-bin visual-word count histogram per pyramid region."""
	histogram, _ = assign_features_to_words(region_descriptors, centers)
	return histogram


def extract_phow(image: Image.Image, centers: np.ndarray, settings: CodebookSettings) -> np.ndarray:
	_, _, region_descriptors = extract_pyramid_sift(image, settings)
	return compute_histograms(region_descriptors, centers)


def build_codebook(
	training_folder: Path, settings: CodebookSettings = CodebookSettings(),
) -> tuple[MiniBatchKMeans, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
	"""Learn training-only words, then encode each training image with that vocabulary."""
	paths = []
	labels = []
	features = []
	image_keypoints = []
	for label in CLASSES:
		# Labels come from folder names, not from filenames or K-means predictions.
		class_folder = training_folder / label
		if not class_folder.is_dir():
			raise ValueError(f"Missing training class folder: {class_folder}")
		class_paths = sorted(path for path in class_folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
		if not class_paths:
			raise ValueError(f"No training images found for {label}.")
		for path in class_paths:
			with Image.open(path) as image:
				_, keypoints, descriptors = extract_pyramid_sift(image, settings)
			paths.append(str(path.resolve()))
			labels.append(label)
			features.append(descriptors)
			image_keypoints.append(keypoints)
	descriptors = np.vstack([region for image_regions in features for region in image_regions])
	if len(descriptors) < settings.words:
		raise ValueError(f"At least {settings.words} training descriptors are needed for the codebook.")
	# The actual visual words are these K centroid vectors, shape (K, 128).
	model = MiniBatchKMeans(n_clusters=settings.words, random_state=42, n_init=3, batch_size=1024).fit(descriptors)
	# Find the closest training descriptor to each centroid to display
	indices, _ = pairwise_distances_argmin_min(model.cluster_centers_, descriptors)
	offsets = np.concatenate(([0], np.cumsum([len(points) for points in image_keypoints])))
	patches = []
	for index in indices:
		# Locate the source image, then the matching keypoint inside that image.
		image_index = int(np.searchsorted(offsets, index, side="right") - 1)
		keypoint = image_keypoints[image_index][index - offsets[image_index]]
		with Image.open(paths[image_index]) as image:
			gray = prepare_image(image, settings.max_image_size)
		region, (left, top) = split_pyramid(gray, settings.pyramid_levels)[keypoint.class_id]
		# Show the nominal four-bin-wide region, not the keypoint diameter.
		size = int(round(6 * keypoint.size))
		point = (keypoint.pt[0] - left, keypoint.pt[1] - top)
		patch = cv2.getRectSubPix(region, (size, size), point)
		patches.append(cv2.resize(patch, (48, 48), interpolation=cv2.INTER_AREA))
	train_histograms = np.vstack([compute_histograms(regions, model.cluster_centers_) for regions in features])
	return model, np.array(patches), np.array(labels, dtype=str), np.array(paths, dtype=str), len(descriptors), train_histograms


def render_codebook(patches: np.ndarray) -> Image.Image:
	"""Display one representative grayscale patch per visual word, in word order."""
	count = len(patches)
	if count < 1:
		raise ValueError("Display at least one visual word.")
	columns = min(5, count)
	rows = (count + columns - 1) // columns
	figure = Figure(figsize=(2 * columns, 2.25 * rows), dpi=100, layout="constrained")
	axes = figure.subplots(rows, columns, squeeze=False)
	for index, axis in enumerate(axes.flat):
		if index < count:
			axis.imshow(patches[index], cmap="gray", vmin=0, vmax=255)
			axis.set_title(f"VW {index}")
		axis.set_axis_off()
	return figure_to_image(figure)


class CodebookConstructionTool(ForensicsTool):
	tool_id = "codebook_construction"
	title = "Codebook construction"
	category = "Set 7"
	description = "Build a visual codebook and configurable spatial-pyramid histograms using dense SIFT."
	requires_image = False

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		folder_name = filedialog.askdirectory(
			parent=parent, title="Choose training image folder", initialdir=str(DEFAULT_DATA / "train"),
		)
		if not folder_name:
			return None
		settings = choose_codebook_settings(parent, title=self.title)
		if settings is None:
			return None
		training_folder = Path(folder_name)
		model, patches, labels, paths, descriptor_count, train_histograms = build_codebook(training_folder, settings)
		output_image = render_codebook(patches)
		output_folder = DEFAULT_DATA / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
		output_folder.mkdir(parents=True)
		model_path = output_folder / "codebook.npz"
		# centers: words used for later nearest-centroid histogram assignment.
		# patches: visualization only. train_labels/train_paths: image metadata.
		np.savez_compressed(
			model_path,
			centers=model.cluster_centers_, patches=patches, train_labels=labels, train_paths=paths,
			train_histograms=train_histograms, pyramid_levels=np.array(settings.pyramid_levels),
			step=np.array(settings.step), sift_size=np.array(settings.sift_size),
			max_image_size=np.array(settings.max_image_size), sigma_min=np.array(settings.sigma_min),
			n_octaves=np.array(settings.n_octaves), n_scales=np.array(settings.n_scales),
			descriptor_method=np.array("spatial_pyramid_dense_sift_v3"),
		)
		return ToolResult(
			image=output_image,
			message=f"Built {settings.words} visual words and {train_histograms.shape[1]}-value PHOW vectors for {len(paths)} training images. Saved codebook to {model_path}.",
			details={
				"Operation": "PHOW codebook construction",
				"Training folder": folder_name,
				"Training images": len(paths),
				"Classes": ", ".join(CLASSES),
				"Descriptors": descriptor_count,
				"Descriptor": "Spatial-pyramid dense SIFT",
				"Pyramid levels": str(settings.pyramid_levels),
				"PHOW dimensions": train_histograms.shape[1],
				"Grid step": settings.step,
				"SIFT size": settings.sift_size,
				"Maximum image size": settings.max_image_size,
				"SIFT minimum sigma": settings.sigma_min,
				"SIFT octaves": settings.n_octaves,
				"SIFT scales per octave": settings.n_scales,
				"Clustering": "MiniBatchKMeans",
				"Visual words (K)": settings.words,
				"Visualization": "Nearest training patch to each centroid",
				"Model": str(model_path),
				"Output mode": output_image.mode,
			},
		)