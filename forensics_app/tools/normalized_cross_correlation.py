"""Find image regions that resemble a selected template."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from skimage import color, feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .constants import OPEN_TYPES


class NormalizedCrossCorrelationTool(ForensicsTool):
	tool_id = "normalized_cross_correlation"
	title = "Normalized cross-correlation"
	category = "Set 4"
	description = "Locate regions resembling a selected template using normalized cross-correlation."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		filename = filedialog.askopenfilename(
			title="Choose template image",
			filetypes=OPEN_TYPES,
		)
		if not filename:
			return None

		with Image.open(filename) as source_template:
			template = np.array(source_template)
		image = np.array(document.current)
		if template.shape[1] > image.shape[1] or template.shape[0] > image.shape[0]:
			raise ValueError("The template must not be larger than the current image.")

		if image.ndim == 3:
			image = color.rgb2gray(image[..., :3])
		if template.ndim == 3:
			template = color.rgb2gray(template[..., :3])
		corr_tmpl = feature.match_template(image, template, pad_input=True)
		figure = Figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
		axis = figure.add_axes((0, 0, 1, 1))

		axis.imshow(corr_tmpl, cmap="gray")
		axis.axis("off")
		canvas = FigureCanvasAgg(figure)
		canvas.draw()
		output_image = Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())

		details = {
			"Operation": "Normalized cross-correlation image",
			"Template": Path(filename).name,
			"Output mode": output_image.mode,
		}
		return ToolResult(
			image=output_image,
			message=f"Generated normalized cross-correlation image using {Path(filename).name}.",
			details=details,
		)