"""Create binary or pixel-preserving threshold masks."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image, ImageOps

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .constants import MASK_MESSAGES, MASK_OPERATIONS
from .utils import choose_masking


class MaskingTool(ForensicsTool):
	tool_id = "masking"
	title = "Masking"
	category = "Image processing"
	description = "Create a binary mask or retain pixels above or below a chosen threshold."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		selection = choose_masking(
			parent,
			title=self.title,
			prompt="Choose masking settings:",
		)
		if selection is None:
			return None

		# Read image
		image = document.current
		image_numpy = np.array(image)
		# Create mask
		mask = np.array(ImageOps.grayscale(image)) > selection.threshold
		if image_numpy.ndim == 3:
			mask = mask[:, :, np.newaxis]
		selected_mask = mask if selection.keep_bright else ~mask
		# Apply mask to image
		if selection.binary:
			image_numpy = selected_mask.astype(np.uint8) * 255
		selected_image = ~image_numpy if selection.invert_image else image_numpy
		image_numpy = selected_image * selected_mask
		# Just keep the first channel for binary output
		if selection.binary and image_numpy.ndim == 3:
			image_numpy = image_numpy[:, :, 0]
		output_image = Image.fromarray(image_numpy)

		mode = "binary" if selection.binary else "bright" if selection.keep_bright else "dark"
		operation = MASK_OPERATIONS[mode]
		message = MASK_MESSAGES[mode].format(
			threshold=selection.threshold,
			region="bright" if selection.keep_bright else "dark",
		)
		return ToolResult(
			image=output_image,
			message=message,
			details={
				"Operation": operation,
				"Threshold": selection.threshold,
				"Binary output": selection.binary,
				"Retained pixels": "Bright" if selection.keep_bright else "Dark",
				"Image inverted": selection.invert_image,
				"Output mode": output_image.mode,
			},
		)