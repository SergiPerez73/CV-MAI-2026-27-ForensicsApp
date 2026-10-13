"""Show Canny edges over the original image."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utils import choose_edge_visualization


class EdgeVisualizationTool(ForensicsTool):
	tool_id = "edge_visualization"
	title = "Edge visualization"
	category = "Image processing"
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
		canvas = FigureCanvasAgg(figure)
		canvas.draw()
		output_image = Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
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
