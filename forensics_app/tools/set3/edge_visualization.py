"""Show Canny edges over the original image."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from skimage import feature

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utils import add_dialog_buttons, add_numeric_field, create_dialog, figure_to_image, wait_for_dialog


@dataclass(frozen=True)
class EdgeVisualizationSettings:
	sigma: float
	low_threshold: float
	high_threshold: float
	line_width: float


def choose_edge_visualization(parent: tk.Misc, *, title: str) -> EdgeVisualizationSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: EdgeVisualizationSettings | None = None
	sigma_value = tk.StringVar(master=dialog, value="2.0")
	low_value = tk.StringVar(master=dialog, value="0.1")
	high_value = tk.StringVar(master=dialog, value="0.2")
	line_width_value = tk.StringVar(master=dialog, value="0.8")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			sigma = float(sigma_value.get())
			low_threshold = float(low_value.get())
			high_threshold = float(high_value.get())
			line_width = float(line_width_value.get())
		except ValueError:
			error_text.set("Enter valid numeric values.")
			return
		if sigma <= 0 or line_width <= 0 or not 0 <= low_threshold < high_threshold <= 1:
			error_text.set("Use positive sigma and width, and 0 <= low < high <= 1.")
			return
		selected = EdgeVisualizationSettings(sigma, low_threshold, high_threshold, line_width)
		dialog.destroy()

	for label, value in (
		("Sigma:", sigma_value),
		("Low threshold (0-1):", low_value),
		("High threshold (0-1):", high_value),
		("Contour width:", line_width_value),
	):
		add_numeric_field(dialog, label, value)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


class EdgeVisualizationTool(ForensicsTool):
	tool_id = "edge_visualization"
	title = "Edge visualization"
	category = "Set 3"
	description = "Overlay Canny edge contours in red on the original image."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		settings = choose_edge_visualization(parent, title=self.title)
		if settings is None:
			return None

		image = document.current.convert("RGB")
		grayscale = np.array(image.convert("L"), dtype=np.float32) / 255
		edges = feature.canny(
			grayscale,
			sigma=settings.sigma,
			low_threshold=settings.low_threshold,
			high_threshold=settings.high_threshold,
		)
		figure = Figure(figsize=(image.width / 100, image.height / 100), dpi=100)
		axis = figure.add_axes((0, 0, 1, 1))
		axis.imshow(image)
		axis.contour(edges, colors="red", linewidths=settings.line_width)
		axis.axis("off")
		output_image = figure_to_image(figure)
		return ToolResult(
			image=output_image,
			message="Displayed Canny edge contours over the original image.",
			details={
				"Operation": "Canny contour overlay",
				"Sigma": settings.sigma,
				"Low threshold": settings.low_threshold,
				"High threshold": settings.high_threshold,
				"Contour width": settings.line_width,
			},
		)
