"""Detect image edges with configurable Canny thresholds."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utils import choose_canny


class CannyEdgeDetectionTool(ForensicsTool):
	tool_id = "canny_edge_detection"
	title = "Canny edge detection"
	category = "Image processing"
	description = "Detect edges with configurable smoothing and hysteresis thresholds."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		settings = choose_canny(parent, title=self.title)
		if settings is None:
			return None

		image = np.array(document.current.convert("L"), dtype=np.float32) / 255
		edges = feature.canny(
			image,
			sigma=settings.sigma,
			low_threshold=settings.low_threshold,
			high_threshold=settings.high_threshold,
		)
		output_image = Image.fromarray(edges.astype(np.uint8) * 255)
		return ToolResult(
			image=output_image,
			message="Detected edges with the Canny filter.",
			details={
				"Operation": "Canny edge detection",
				"Sigma": settings.sigma,
				"Low threshold": settings.low_threshold,
				"High threshold": settings.high_threshold,
			},
		)
