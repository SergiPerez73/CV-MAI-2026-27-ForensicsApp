"""Build and visualize an eigenface model from labeled face images."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utils import add_dialog_buttons, add_numeric_field, create_dialog, figure_to_image, wait_for_dialog
from .eigenfaces import fit_eigenfaces, load_face_folder, save_eigenface_dataset


DEFAULT_DATA = Path(__file__).resolve().parents[3] / "sets" / "Set6_images" / "images"


@dataclass(frozen=True)
class EigenfaceSettings:
	components: int = 20
	width: int = 37
	height: int = 50


def choose_eigenfaces(parent: tk.Misc, *, title: str) -> EigenfaceSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: EigenfaceSettings | None = None
	fields = (
		("PCA components (K):", "20"),
		("Face width (px):", "37"),
		("Face height (px):", "50"),
	)
	values = [tk.StringVar(master=dialog, value=default) for _, default in fields]
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			settings = EigenfaceSettings(*(int(value.get()) for value in values))
			if min(settings.components, settings.width, settings.height) < 1:
				raise ValueError("K, width, and height must be positive.")
		except ValueError as error:
			error_text.set(str(error))
			return
		selected = settings
		dialog.destroy()

	for (label, _), value in zip(fields, values):
		add_numeric_field(dialog, label, value)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


def render_eigenfaces(components: np.ndarray, shape: tuple[int, int]) -> Image.Image:
	count = len(components)
	if count < 1:
		raise ValueError("Display at least one eigenface.")
	columns = min(4, count)
	rows = (count + columns - 1) // columns
	figure = Figure(figsize=(2 * columns, 2.5 * rows), dpi=100, layout="constrained")
	axes = figure.subplots(rows, columns, squeeze=False)
	for index, axis in enumerate(axes.flat):
		if index < count:
			# PCA components contain signed weights, not ordinary image intensities.
			# Normalize each component for display only; save the original weights.
			axis.imshow(components[index].reshape(shape), cmap="gray")
			axis.set_title(f"Eigenface {index + 1}")
		axis.set_axis_off()
	return figure_to_image(figure)


class EigenfaceBuilderTool(ForensicsTool):
	tool_id = "eigenface_builder"
	title = "Eigenface builder"
	category = "Set 6"
	description = "Build a PCA face model with configurable K and visualize its eigenfaces."
	requires_image = False

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		train_name = filedialog.askdirectory(
			parent=parent, title="Choose training image folder", initialdir=str(DEFAULT_DATA / "train_data/faces"),
		)
		if not train_name:
			return None
		test_name = filedialog.askdirectory(
			parent=parent, title="Choose test image folder", initialdir=str(DEFAULT_DATA / "test_data/faces"),
		)
		if not test_name:
			return None
		settings = choose_eigenfaces(parent, title=self.title)
		if settings is None:
			return None
		output_folder = DEFAULT_DATA / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
		size = (settings.width, settings.height)
		train_images, train_labels, train_paths = load_face_folder(Path(train_name), size)
		_, test_labels, test_paths = load_face_folder(Path(test_name), size)
		if set(train_paths) & set(test_paths):
			raise ValueError("Training and test folders must not contain the same image paths.")
		# Fit only on training faces: the test set must not influence the mean or basis.
		pca = fit_eigenfaces(train_images, settings.components)
		output_image = render_eigenfaces(pca.components_, (settings.height, settings.width))
		model_path = save_eigenface_dataset(
			output_folder, train_images, train_labels, train_paths, test_labels, test_paths, pca,
		)
		variance = float(pca.explained_variance_ratio_.sum() * 100)
		return ToolResult(
			image=output_image,
			message=f"Built {settings.components} eigenfaces ({variance:.1f}% variance). Saved model to {model_path}.",
			details={
				"Operation": "Eigenface builder (PCA)",
				"Training folder": train_name,
				"Test folder": test_name,
				"Training images": len(train_paths),
				"Test images": len(test_paths),
				"Components (K)": settings.components,
				"Eigenfaces displayed": settings.components,
				"Face size": f"{settings.width} x {settings.height}",
				"Explained variance": f"{variance:.2f}%",
				"Model": str(model_path),
				"Labels": str(output_folder / "labels.json"),
				"Output mode": output_image.mode,
			},
		)