"""Find a rotated template in the current image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from skimage import color, feature, transform

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..constants import OPEN_TYPES
from ..utils import add_dialog_buttons, add_numeric_field, create_dialog, figure_to_image, wait_for_dialog


@dataclass(frozen=True)
class TemplateMatchingSettings:
	angle: float
	threshold: float


def choose_template_matching(parent: tk.Misc, *, title: str) -> TemplateMatchingSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: TemplateMatchingSettings | None = None
	angle_value = tk.StringVar(master=dialog, value="0")
	threshold_value = tk.StringVar(master=dialog, value="0.8")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			angle = float(angle_value.get())
			threshold = float(threshold_value.get())
		except ValueError:
			error_text.set("Enter valid numeric values.")
			return
		if not -1 <= threshold <= 1:
			error_text.set("Use a threshold from -1 to 1.")
			return
		selected = TemplateMatchingSettings(angle, threshold)
		dialog.destroy()

	add_numeric_field(dialog, "Angle (degrees):", angle_value)
	add_numeric_field(dialog, "Threshold (-1 to 1):", threshold_value)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


class TemplateMatchingTool(ForensicsTool):
	tool_id = "template_matching"
	title = "Template matching"
	category = "Set 4"
	description = "Find a template after rotating it by a selected angle."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		filename = filedialog.askopenfilename(title="Choose template image", filetypes=OPEN_TYPES)
		if not filename:
			return None

		settings = choose_template_matching(parent, title=self.title)
		if settings is None:
			return None

		with Image.open(filename) as source_template:
			template = np.array(source_template)
		image = np.array(document.current)
		if image.ndim == 3:
			image = color.rgb2gray(image[..., :3])
		if template.ndim == 3:
			template = color.rgb2gray(template[..., :3])
		template_r = transform.rotate(template, settings.angle)
		if template_r.shape[1] > image.shape[1] or template_r.shape[0] > image.shape[0]:
			raise ValueError("The rotated template must not be larger than the current image.")
		corr_tmpl = feature.match_template(image, template_r, pad_input=True)
		y_coords, x_coords = np.where(corr_tmpl > settings.threshold)
		figure = Figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
		axis = figure.add_axes((0, 0, 1, 1))
		axis.imshow(image * 0.4, cmap="gray")
		axis.plot(x_coords, y_coords, "r.", markersize=6)
		axis.axis("off")
		output_image = figure_to_image(figure)

		return ToolResult(
			image=output_image,
			message=f"Matched rotated template {Path(filename).name}.",
			details={
				"Operation": "Rotated template matching",
				"Template": Path(filename).name,
				"Angle": settings.angle,
				"Threshold": settings.threshold,
				"Output mode": output_image.mode,
			},
		)