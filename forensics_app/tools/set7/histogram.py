"""Show visual-word frequencies for the currently opened image."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch
from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utils import figure_to_image
from .codebook import CodebookSettings, DEFAULT_DATA, assign_features_to_words, extract_pyramid_sift, load_codebook


def choose_codebook(parent: tk.Misc) -> str:
	return filedialog.askopenfilename(
		parent=parent, title="Choose visual-word model", initialdir=str(DEFAULT_DATA),
		filetypes=[("Visual-word model", "*.npz")],
	)


def render_histogram(
	image: Image.Image, histogram: np.ndarray, settings: CodebookSettings, prediction: str | None = None,
	*, positions: np.ndarray | None = None, assignments: np.ndarray | None = None,
) -> Image.Image:
	frequencies = histogram / histogram.sum() if histogram.sum() else histogram.copy()
	word_frequencies = frequencies.reshape(-1, settings.words).sum(axis=0)
	palette = colormaps["tab20" if settings.words <= 20 else "turbo"].resampled(settings.words)
	colors = palette(np.arange(settings.words))
	figure = Figure(figsize=(10, 6), dpi=100, layout="constrained")
	grid = figure.add_gridspec(2, 2, width_ratios=(1, 2))
	image_axis = figure.add_subplot(grid[:, 0])
	image_axis.imshow(image.convert("RGB"))
	image_axis.set_title(prediction or "Opened image", wrap=True)
	if positions is not None and assignments is not None:
		if len(assignments):
			image_axis.set_title("Features assigned to nearest visual word", wrap=True)
		else:
			image_axis.set_title("No SIFT features", wrap=True)
	image_axis.set_axis_off()
	word_axis = figure.add_subplot(grid[0, 1])
	word_axis.bar(np.arange(settings.words), word_frequencies, color=colors, edgecolor="black")
	word_axis.set(title="Histogram of visual words", xlabel="Visual word index", ylabel="Normalized frequency")
	if settings.words <= 20:
		word_axis.set_xticks(np.arange(settings.words))
	if positions is not None and assignments is not None:
		for position, word in zip(positions, assignments):
			connection = ConnectionPatch(
				xyA=position, xyB=(word, word_frequencies[word]),
				coordsA="data", coordsB="data", axesA=image_axis, axesB=word_axis,
				color=colors[word], linewidth=0.5, alpha=0.35, clip_on=False,
			)
			connection.set_in_layout(False)
			figure.add_artist(connection)
	pyramid_axis = figure.add_subplot(grid[1, 1])
	pyramid_axis.bar(
		np.arange(len(frequencies)), frequencies, width=1,
		color=colors[np.arange(len(frequencies)) % settings.words],
	)
	pyramid_axis.set(
		title=f"Spatial pyramid ({len(frequencies)} bins)",
		xlabel="Region x visual word index", ylabel="Normalized frequency",
	)
	for boundary in np.cumsum(settings.pyramid_levels)[:-1] * settings.words:
		pyramid_axis.axvline(boundary - 0.5, color="black", linewidth=0.8)
	return figure_to_image(figure)


class VisualWordHistogramTool(ForensicsTool):
	tool_id = "visual_word_histogram"
	title = "Visual word histogram"
	category = "Set 7"
	description = "Show nearest-centroid feature assignments and matching visual-word histograms."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		model_name = choose_codebook(parent)
		if not model_name:
			return None
		model = load_codebook(Path(model_name))
		gray, keypoints, descriptors = extract_pyramid_sift(document.current, model.settings)
		histogram, assignments = assign_features_to_words(descriptors, model.centers)
		positions = np.array([point.pt for point in keypoints], dtype=float).reshape(-1, 2)
		positions *= (document.current.width / gray.shape[1], document.current.height / gray.shape[0])
		output_image = render_histogram(
			document.current, histogram, model.settings, positions=positions, assignments=assignments,
		)
		return ToolResult(
			image=output_image,
			message=f"Showing {model.settings.words} visual words and {len(histogram)} spatial-pyramid bins.",
			details={
				"Operation": "Visual word histogram", "Model": model_name,
				"Test image": "Currently open image", "Visual words (K)": model.settings.words,
				"Pyramid levels": str(model.settings.pyramid_levels), "PHOW dimensions": len(histogram),
				"Descriptor assignments": int(histogram.sum()), "Normalization": "L1 frequency",
				"Assignment": "Nearest centroid (Euclidean distance)",
				"Visualization": "Features connected to assigned visual-word bars with matching colors",
			},
		)