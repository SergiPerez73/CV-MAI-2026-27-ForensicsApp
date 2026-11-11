"""Load CNN feature vectors and visualize a selected t-SNE or UMAP projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure
from sklearn.manifold import TSNE

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utils import add_dialog_buttons, add_numeric_field, create_dialog, figure_to_image, wait_for_dialog
from .classification import DEFAULT_DATA, log_progress
from .features import FeatureVectors, load_features


@dataclass(frozen=True)
class ProjectionSettings:
	method: str = "tsne"
	perplexity: float = 30.0
	iterations: int = 1000
	n_neighbors: int = 15
	min_dist: float = 0.1
	seed: int = 42

	def __post_init__(self) -> None:
		if self.method not in {"tsne", "umap"}:
			raise ValueError("Choose t-SNE or UMAP.")
		if not np.isfinite(self.perplexity) or self.perplexity <= 0 or self.iterations < 250:
			raise ValueError("t-SNE needs positive perplexity and at least 250 iterations.")
		if self.n_neighbors < 2 or not np.isfinite(self.min_dist) or not 0 <= self.min_dist <= 1:
			raise ValueError("UMAP needs at least two neighbors and a minimum distance between 0 and 1.")
		if not 0 <= self.seed < 2 ** 32:
			raise ValueError("Random seed must be between 0 and 2**32 - 1.")


def project_features(vectors: FeatureVectors, settings: ProjectionSettings = ProjectionSettings()) -> np.ndarray:
	count = len(vectors.features)
	log_progress(f"Preparing {settings.method} projection for {count} feature vectors...")
	if count < 3:
		raise ValueError("At least three feature vectors are needed for a 2D projection.")
	if settings.method == "tsne":
		if settings.perplexity >= count:
			raise ValueError("t-SNE perplexity must be smaller than the number of feature vectors.")
		iteration_key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
		projector = TSNE(
			n_components=2, perplexity=settings.perplexity, random_state=settings.seed, learning_rate="auto",
			init="pca" if vectors.features.shape[1] >= 2 else "random", verbose=1, **{iteration_key: settings.iterations},
		)
	else:
		if settings.n_neighbors >= count:
			raise ValueError("UMAP neighbors must be smaller than the number of feature vectors.")
		try:
			log_progress("Loading UMAP runtime; first use may compile Numba kernels...")
			from umap import UMAP
		except ImportError as error:
			raise ValueError("UMAP needs umap-learn. Install it with: python -m pip install umap-learn") from error
		projector = UMAP(
			n_components=2, n_neighbors=settings.n_neighbors, min_dist=settings.min_dist,
			random_state=settings.seed, init="random", n_jobs=1, verbose=True,
		)
	log_progress(f"Fitting {settings.method} 2D projection...")
	coordinates = np.array(projector.fit_transform(vectors.features), dtype=np.float32)
	if coordinates.shape != (count, 2) or not np.isfinite(coordinates).all():
		raise ValueError("The projection did not produce finite 2D coordinates.")
	log_progress(f"{settings.method} projection completed.")
	return coordinates


def render_projection(coordinates: np.ndarray, vectors: FeatureVectors, method: str):
	log_progress("Rendering labeled feature projection...")
	figure = Figure(figsize=(9, 6), dpi=100, layout="constrained")
	axis = figure.subplots()
	palette = colormaps["tab10" if len(vectors.class_names) <= 10 else "turbo"].resampled(len(vectors.class_names))
	for label, name in enumerate(vectors.class_names):
		selected = vectors.labels == label
		if selected.any():
			axis.scatter(coordinates[selected, 0], coordinates[selected, 1], s=12, alpha=0.7, color=palette(label), label=name)
	title = "t-SNE" if method == "tsne" else "UMAP"
	axis.set(title=f"{title}: test-set CNN features", xlabel=f"{title} 1", ylabel=f"{title} 2")
	axis.legend(loc="center left", bbox_to_anchor=(1, 0.5))
	return figure_to_image(figure)


def choose_projection_settings(parent: tk.Misc, count: int) -> ProjectionSettings | None:
	dialog = create_dialog(parent, title="Feature projection")
	selected: ProjectionSettings | None = None
	method = tk.StringVar(master=dialog, value="t-SNE")
	fields = (
		("Perplexity:", str(min(30.0, max(1.0, (count - 1) / 3)))), ("Iterations:", "1000"),
		("Neighbors:", str(min(15, max(2, count - 1)))), ("Minimum distance:", "0.1"),
	)
	values = [tk.StringVar(master=dialog, value=value) for _, value in fields]
	error_text = tk.StringVar(master=dialog)
	row = ttk.Frame(dialog)
	row.pack(fill="x", padx=16, pady=(12, 0))
	ttk.Label(row, text="Projection:").pack(side="left")
	selector = ttk.Combobox(row, textvariable=method, values=["t-SNE", "UMAP"], state="readonly", width=15)
	selector.pack(side="right")
	controls = [add_numeric_field(dialog, label, value) for (label, _), value in zip(fields, values)]

	def update_method(_event=None) -> None:
		for index, control in enumerate(controls):
			active = (index < 2) == (method.get() == "t-SNE")
			control.configure(state="normal" if active else "disabled")

	def apply_settings() -> None:
		nonlocal selected
		try:
			settings = ProjectionSettings(
				method="tsne" if method.get() == "t-SNE" else "umap", perplexity=float(values[0].get()),
				iterations=int(values[1].get()), n_neighbors=int(values[2].get()), min_dist=float(values[3].get()),
			)
			if count < 3 or (settings.method == "tsne" and settings.perplexity >= count) or (settings.method == "umap" and settings.n_neighbors >= count):
				raise ValueError("Need at least three vectors, with perplexity/neighbors smaller than the sample count.")
		except ValueError as error:
			error_text.set(str(error))
			return
		selected = settings
		dialog.destroy()

	selector.bind("<<ComboboxSelected>>", update_method)
	update_method()
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


class CNNFeatureProjectionTool(ForensicsTool):
	tool_id = "cnn_feature_projection"
	title = "Project feature vectors"
	category = "Set 8"
	description = "Load test feature vectors and visualize a selected t-SNE or UMAP projection."
	requires_image = False

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		file_name = filedialog.askopenfilename(
			parent=parent, title="Choose saved CNN test features", initialdir=str(DEFAULT_DATA),
			filetypes=[("CNN feature vectors", "*.npz")],
		)
		if not file_name:
			return None
		vectors = load_features(Path(file_name))
		settings = choose_projection_settings(parent, len(vectors.features))
		if settings is None:
			return None
		coordinates = project_features(vectors, settings)
		return ToolResult(
			image=render_projection(coordinates, vectors, settings.method), message=f"Projected {len(coordinates)} test features using {settings.method}.",
			details={
				"Operation": "CNN feature projection", "Feature archive": file_name, "Samples": len(coordinates),
				"Feature dimensions": vectors.features.shape[1], "Class names": ", ".join(vectors.class_names),
				"Color labels": "True test classes, not model predictions", "Feature layer": vectors.metadata.get("feature_layer"),
				**asdict(settings),
			},
		)