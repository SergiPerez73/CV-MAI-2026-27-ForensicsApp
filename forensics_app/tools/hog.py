"""Extract and visualize Histogram of Oriented Gradients features."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from skimage import color, feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utils import add_dialog_buttons, add_numeric_field, create_dialog, wait_for_dialog


@dataclass(frozen=True)
class HOGSettings:
	orientations: int
	pixels_per_cell: int
	cells_per_block: int


def choose_hog(parent: tk.Misc, *, title: str) -> HOGSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: HOGSettings | None = None
	orientations_value = tk.StringVar(master=dialog, value="9")
	pixels_per_cell_value = tk.StringVar(master=dialog, value="8")
	cells_per_block_value = tk.StringVar(master=dialog, value="3")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			orientations = int(orientations_value.get())
			pixels_per_cell = int(pixels_per_cell_value.get())
			cells_per_block = int(cells_per_block_value.get())
		except ValueError:
			error_text.set("Enter positive whole numbers.")
			return
		if min(orientations, pixels_per_cell, cells_per_block) < 1:
			error_text.set("Enter positive whole numbers.")
			return
		selected = HOGSettings(orientations, pixels_per_cell, cells_per_block)
		dialog.destroy()

	add_numeric_field(dialog, "Orientations:", orientations_value)
	add_numeric_field(dialog, "Pixels per cell:", pixels_per_cell_value)
	add_numeric_field(dialog, "Cells per block:", cells_per_block_value)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


class HOGTool(ForensicsTool):
	tool_id = "hog"
	title = "HOG extraction"
	category = "Set 4"
	description = "Extract and visualize Histogram of Oriented Gradients features."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		settings = choose_hog(parent, title=self.title)
		if settings is None:
			return None

		image = np.array(document.current)
		if image.ndim == 3:
			image = color.rgb2gray(image[..., :3])
		_, hog_image = feature.hog(
			image,
			orientations=settings.orientations,
			pixels_per_cell=(settings.pixels_per_cell, settings.pixels_per_cell),
			cells_per_block=(settings.cells_per_block, settings.cells_per_block),
			visualize=True,
		)
		figure = Figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
		axis = figure.add_axes((0, 0, 1, 1))
		axis.imshow(hog_image, cmap="gray")
		axis.axis("off")
		canvas = FigureCanvasAgg(figure)
		canvas.draw()
		output_image = Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())

		return ToolResult(
			image=output_image,
			message="Generated HOG visualization.",
			details={
				"Operation": "HOG visualization",
				"Orientations": settings.orientations,
				"Pixels per cell": f"{settings.pixels_per_cell} x {settings.pixels_per_cell}",
				"Cells per block": f"{settings.cells_per_block} x {settings.cells_per_block}",
				"Output mode": output_image.mode,
			},
		)