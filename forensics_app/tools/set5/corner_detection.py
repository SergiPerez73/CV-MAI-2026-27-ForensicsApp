"""Detect FAST or Harris corners and visualize their peaks."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from skimage import color, feature

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from .utils import render_keypoints
from ..utils import add_dialog_buttons, add_numeric_field, create_dialog, figure_to_image, wait_for_dialog


@dataclass(frozen=True)
class CornerSettings:
	detector: str
	peak_threshold: float
	output: str = "peaks"


def render_response(response: np.ndarray) -> Image.Image:
	"""Render a FAST or Harris response image without peak markers."""
	figure = Figure(figsize=(response.shape[1] / 100, response.shape[0] / 100), dpi=100)
	axis = figure.add_axes((0, 0, 1, 1))
	axis.imshow(response, cmap="gray")
	axis.axis("off")
	return figure_to_image(figure)


def choose_corner_detection(parent: tk.Misc, *, title: str) -> CornerSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: CornerSettings | None = None
	detector_value = tk.StringVar(master=dialog, value="fast")
	threshold_value = tk.StringVar(master=dialog, value="0.1")
	output_value = tk.StringVar(master=dialog, value="peaks")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			threshold = float(threshold_value.get())
		except ValueError:
			error_text.set("Enter a numeric threshold.")
			return
		if not 0 <= threshold <= 1:
			error_text.set("Use a threshold from 0 to 1.")
			return
		selected = CornerSettings(detector_value.get(), threshold, output_value.get())
		dialog.destroy()

	ttk.Label(dialog, text="Detector:").pack(anchor="w", padx=16, pady=(12, 0))
	ttk.Radiobutton(dialog, text="FAST", variable=detector_value, value="fast").pack(anchor="w", padx=16)
	ttk.Radiobutton(dialog, text="Harris", variable=detector_value, value="harris").pack(anchor="w", padx=16)
	ttk.Label(dialog, text="Output:").pack(anchor="w", padx=16, pady=(12, 0))
	ttk.Radiobutton(dialog, text="Detected peaks", variable=output_value, value="peaks").pack(anchor="w", padx=16)
	ttk.Radiobutton(dialog, text="Detector response", variable=output_value, value="response").pack(anchor="w", padx=16)
	add_numeric_field(dialog, "Peak threshold (relative):", threshold_value)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


class CornerDetectionTool(ForensicsTool):
	tool_id = "corner_detection"
	title = "Corner detection"
	category = "Set 5"
	description = "Detect FAST or Harris corners with a configurable peak threshold."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		settings = choose_corner_detection(parent, title=self.title)
		if settings is None:
			return None

		image = np.array(document.current)
		image = image if image.ndim == 2 else color.rgb2gray(image[..., :3])
		response = feature.corner_fast(image) if settings.detector == "fast" else feature.corner_harris(image)
		detector = "FAST" if settings.detector == "fast" else "Harris"
		if settings.output == "response":
			output_image = render_response(response)
			return ToolResult(
				image=output_image,
				message=f"Generated {detector} corner response.",
				details={
					"Operation": f"{detector} corner response",
					"Output mode": output_image.mode,
				},
			)

		keypoints = feature.corner_peaks(response, threshold_rel=settings.peak_threshold)
		output_image = render_keypoints(document.current, keypoints)

		return ToolResult(
			image=output_image,
			message=f"Detected {len(keypoints)} {detector} corners.",
			details={
				"Operation": f"{detector} corner detection",
				"Peak threshold": settings.peak_threshold,
				"Corners": len(keypoints),
				"Output mode": output_image.mode,
			},
		)