"""Apply a mean convolution with a selectable kernel shape."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image
from scipy import ndimage

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utils import choose_convolution


class ConvolutionTool(ForensicsTool):
	tool_id = "convolution"
	title = "Convolution"
	category = "Image processing"
	description = "Apply a mean convolution with a selected kernel size and direction."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		settings = choose_convolution(parent, title=self.title)
		if settings is None:
			return None

		image = document.current
		if image.mode != "L":
			image = image.convert("RGB")
		image_numpy = np.array(image, dtype=np.float32)

		if settings.direction == "horizontal":
			kernel_shape = (1, settings.size)
		elif settings.direction == "vertical":
			kernel_shape = (settings.size, 1)
		else:
			kernel_shape = (settings.size, settings.size)
		kernel = np.ones(kernel_shape, dtype=np.float32) / np.prod(kernel_shape)

		if image_numpy.ndim == 2:
			result = ndimage.convolve(image_numpy, kernel)
		else:
			result = np.stack(
				[ndimage.convolve(image_numpy[:, :, channel], kernel) for channel in range(3)],
				axis=2,
			)
		output_image = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
		return ToolResult(
			image=output_image,
			message=f"Applied {settings.direction} mean convolution with size {settings.size}.",
			details={
				"Operation": "Mean convolution",
				"Kernel shape": f"{kernel_shape[0]} x {kernel_shape[1]}",
				"Direction": settings.direction.title(),
				"Output mode": output_image.mode,
			},
		)
