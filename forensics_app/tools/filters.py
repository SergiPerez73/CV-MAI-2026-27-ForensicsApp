"""Apply common smoothing and edge-detection filters."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .constants import FILTER_NAMES
from .utils import choose_filter


class FiltersTool(ForensicsTool):
	tool_id = "filters"
	title = "Filters"
	category = "Image processing"
	description = "Apply Gaussian smoothing, median denoising, or Sobel edge detection."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		settings = choose_filter(parent, title=self.title)
		if settings is None:
			return None

		image = document.current
		is_color = image.mode != "L"
		image = image.convert("RGB" if is_color else "L")
		image_numpy = np.array(image, dtype=np.float32) / 255

		if settings.name == "gaussian":
			result = filters.gaussian(
				image_numpy,
				sigma=settings.sigma,
				channel_axis=-1 if is_color else None,
			)
		elif settings.name == "median":
			if is_color:
				result = np.stack(
					[filters.median(image_numpy[:, :, channel]) for channel in range(3)],
					axis=2,
				)
			else:
				result = filters.median(image_numpy)
		else:
			grayscale = np.array(image.convert("L"), dtype=np.float32) / 255
			result = filters.sobel(grayscale)

		output_image = Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
		label = FILTER_NAMES[settings.name]
		details = {
			"Operation": label,
			"Output mode": output_image.mode,
		}
		if settings.name == "gaussian":
			details["Sigma"] = settings.sigma
		return ToolResult(
			image=output_image,
			message=f"Applied {label.lower()}.",
			details=details,
		)
