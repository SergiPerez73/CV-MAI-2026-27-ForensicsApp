"""Sharpen images with an unsharp mask."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utils import choose_sharpening


class ImageSharpeningTool(ForensicsTool):
	tool_id = "image_sharpening"
	title = "Image sharpening"
	category = "Image processing"
	description = "Sharpen an image with a configurable unsharp mask."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		settings = choose_sharpening(parent, title=self.title)
		if settings is None:
			return None

		image = document.current
		is_color = image.mode != "L"
		image = image.convert("RGB" if is_color else "L")
		image_numpy = np.array(image, dtype=np.float32) / 255
		sharpened = filters.unsharp_mask(
			image_numpy,
			radius=settings.radius,
			amount=settings.amount,
		)
		output_image = Image.fromarray((np.clip(sharpened, 0, 1) * 255).astype(np.uint8))
		return ToolResult(
			image=output_image,
			message="Applied unsharp mask.",
			details={
				"Operation": "Unsharp mask",
				"Radius": settings.radius,
				"Amount": settings.amount,
				"Output mode": output_image.mode,
			},
		)
