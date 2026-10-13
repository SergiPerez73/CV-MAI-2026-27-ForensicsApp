"""Detect image edges with configurable Canny thresholds."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utils import add_dialog_buttons, add_numeric_field, create_dialog, wait_for_dialog


@dataclass(frozen=True)
class CannySettings:
	sigma: float
	low_threshold: float
	high_threshold: float


def choose_canny(parent: tk.Misc, *, title: str) -> CannySettings | None:
	dialog = create_dialog(parent, title=title)
	selected: CannySettings | None = None
	sigma_value = tk.StringVar(master=dialog, value="2.0")
	low_value = tk.StringVar(master=dialog, value="0.1")
	high_value = tk.StringVar(master=dialog, value="0.2")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			sigma = float(sigma_value.get())
			low_threshold = float(low_value.get())
			high_threshold = float(high_value.get())
		except ValueError:
			error_text.set("Enter valid numeric values.")
			return
		if sigma <= 0 or not 0 <= low_threshold < high_threshold <= 1:
			error_text.set("Use sigma > 0 and thresholds with 0 <= low < high <= 1.")
			return
		selected = CannySettings(sigma, low_threshold, high_threshold)
		dialog.destroy()

	for label, value in (
		("Sigma:", sigma_value),
		("Low threshold (0-1):", low_value),
		("High threshold (0-1):", high_value),
	):
		add_numeric_field(dialog, label, value)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


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
