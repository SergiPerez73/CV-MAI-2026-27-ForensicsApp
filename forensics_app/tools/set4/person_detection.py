"""Detect a person template using sliding-window HOG descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..constants import OPEN_TYPES
from ..utils import add_dialog_buttons, add_numeric_field, create_dialog, figure_to_image, wait_for_dialog


STEP_SIZE = 8


@dataclass(frozen=True)
class PersonDetectionSettings:
	orientations: int
	pixels_per_cell: int
	cells_per_block: int
	metric: str


def choose_person_detection(parent: tk.Misc, *, title: str) -> PersonDetectionSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: PersonDetectionSettings | None = None
	orientations_value = tk.StringVar(master=dialog, value="16")
	pixels_per_cell_value = tk.StringVar(master=dialog, value="8")
	cells_per_block_value = tk.StringVar(master=dialog, value="2")
	metric_value = tk.StringVar(master=dialog, value="euclidean")
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
		selected = PersonDetectionSettings(
			orientations,
			pixels_per_cell,
			cells_per_block,
			metric_value.get(),
		)
		dialog.destroy()

	add_numeric_field(dialog, "Orientations:", orientations_value)
	add_numeric_field(dialog, "Pixels per cell:", pixels_per_cell_value)
	add_numeric_field(dialog, "Cells per block:", cells_per_block_value)
	ttk.Label(dialog, text="Comparison:").pack(anchor="w", padx=16, pady=(12, 0))
	for value, label in (("euclidean", "Euclidean distance"), ("dot_product", "Dot product")):
		ttk.Radiobutton(dialog, text=label, variable=metric_value, value=value).pack(anchor="w", padx=16)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


class PersonDetectionTool(ForensicsTool):
	tool_id = "person_detection"
	title = "Person detection (HOG)"
	category = "Set 4"
	description = "Locate a person template with configurable sliding-window HOG comparisons."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		filename = filedialog.askopenfilename(title="Choose person template", filetypes=OPEN_TYPES)
		if not filename:
			return None
		settings = choose_person_detection(parent, title=self.title)
		if settings is None:
			return None

		with Image.open(filename) as source_template:
			template = np.array(source_template.convert("RGB"))
		image = np.array(document.current.convert("RGB"))
		template_height, template_width = template.shape[:2]
		image_height, image_width = image.shape[:2]
		if template_height > image_height or template_width > image_width:
			raise ValueError("The person template must not be larger than the current image.")

		template_descriptor = feature.hog(
			template,
			orientations=settings.orientations,
			pixels_per_cell=(settings.pixels_per_cell, settings.pixels_per_cell),
			cells_per_block=(settings.cells_per_block, settings.cells_per_block),
			channel_axis=-1,
		)
		scores: list[float] = []
		coordinates: list[tuple[int, int]] = []

		for y_coord in range(0, image_height - template_height + 1, STEP_SIZE):
			for x_coord in range(0, image_width - template_width + 1, STEP_SIZE):
				person_slice = image[y_coord:y_coord + template_height, x_coord:x_coord + template_width]
				person_descriptor = feature.hog(
					person_slice,
					orientations=settings.orientations,
					pixels_per_cell=(settings.pixels_per_cell, settings.pixels_per_cell),
					cells_per_block=(settings.cells_per_block, settings.cells_per_block),
					channel_axis=-1,
				)
				coordinates.append((y_coord, x_coord))
				if settings.metric == "euclidean":
					scores.append(float(np.linalg.norm(template_descriptor - person_descriptor)))
				else:
					scores.append(float(np.dot(person_descriptor, template_descriptor)))

		match_index = int(np.argmin(scores) if settings.metric == "euclidean" else np.argmax(scores))
		match_location = coordinates[match_index]
		metric_name = "Euclidean distance" if settings.metric == "euclidean" else "Dot product"
		figure = Figure(figsize=(image_width / 100, image_height / 100), dpi=100)
		axis = figure.add_axes((0, 0, 1, 1))
		y_coord, x_coord = match_location
		axis.imshow(image)
		axis.add_patch(
			Rectangle((x_coord, y_coord), template_width, template_height, fill=False, linewidth=2, edgecolor="red")
		)
		axis.set_axis_off()
		output_image = figure_to_image(figure)

		return ToolResult(
			image=output_image,
			message=f"Detected person template {Path(filename).name}.",
			details={
				"Operation": "HOG person detection",
				"Template": Path(filename).name,
				"Metric": metric_name,
				"Orientations": settings.orientations,
				"Pixels per cell": f"{settings.pixels_per_cell} x {settings.pixels_per_cell}",
				"Cells per block": f"{settings.cells_per_block} x {settings.cells_per_block}",
				"Step size": STEP_SIZE,
				"Location": f"{match_location[1]}, {match_location[0]}",
				"Output mode": output_image.mode,
			},
		)