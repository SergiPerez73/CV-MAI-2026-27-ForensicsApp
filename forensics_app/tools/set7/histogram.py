"""Show visual-word frequencies for the currently opened image."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utils import figure_to_image
from .codebook import CodebookSettings, DEFAULT_DATA, extract_phow, load_codebook


def choose_codebook(parent: tk.Misc) -> str:
	return filedialog.askopenfilename(
		parent=parent, title="Choose visual-word model", initialdir=str(DEFAULT_DATA),
		filetypes=[("Visual-word model", "*.npz")],
	)


def render_histogram(
	image: Image.Image, histogram: np.ndarray, settings: CodebookSettings, prediction: str | None = None,
) -> Image.Image:
	frequencies = histogram / histogram.sum() if histogram.sum() else histogram.copy()
	word_frequencies = frequencies.reshape(-1, settings.words).sum(axis=0)
	figure = Figure(figsize=(10, 6), dpi=100, layout="constrained")
	grid = figure.add_gridspec(2, 2, width_ratios=(1, 2))
	image_axis = figure.add_subplot(grid[:, 0])
	image_axis.imshow(image.convert("RGB"))
	image_axis.set_title(prediction or "Opened image", wrap=True)
	image_axis.set_axis_off()
	word_axis = figure.add_subplot(grid[0, 1])
	word_axis.bar(np.arange(settings.words), word_frequencies, color="gray", edgecolor="black")
	word_axis.set(title="Histogram of visual words", xlabel="Visual word index", ylabel="Normalized frequency")
	pyramid_axis = figure.add_subplot(grid[1, 1])
	pyramid_axis.bar(np.arange(len(frequencies)), frequencies, width=1, color="gray")
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
	description = "Show visual-word and spatial-pyramid histograms of the opened image."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		model_name = choose_codebook(parent)
		if not model_name:
			return None
		model = load_codebook(Path(model_name))
		histogram = extract_phow(document.current, model.centers, model.settings)
		output_image = render_histogram(document.current, histogram, model.settings)
		return ToolResult(
			image=output_image,
			message=f"Showing {model.settings.words} visual words and {len(histogram)} spatial-pyramid bins.",
			details={
				"Operation": "Visual word histogram", "Model": model_name,
				"Test image": "Currently open image", "Visual words (K)": model.settings.words,
				"Pyramid levels": str(model.settings.pyramid_levels), "PHOW dimensions": len(histogram),
				"Descriptor assignments": int(histogram.sum()), "Normalization": "L1 frequency",
			},
		)