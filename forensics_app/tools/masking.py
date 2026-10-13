"""Create binary or pixel-preserving threshold masks."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageOps

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .constants import OPEN_TYPES
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

		filename = filedialog.askopenfilename(
			title="Choose image to mask",
			filetypes=OPEN_TYPES,
		)
		if not filename:
			return None

		with Image.open(filename) as image:
			on_top = image.convert("RGBA").resize(document.current.size)
		image_numpy = np.array(on_top)
		# Create mask
		mask_image = document.current if selection.use_background else on_top
		mask = np.array(ImageOps.grayscale(mask_image)) > selection.threshold
		selected_mask = mask if selection.keep_bright else ~mask
		# Apply mask to the image placed on top
		if selection.binary:
			image_numpy[:, :, :3] = 255
		elif selection.invert_image:
			image_numpy[:, :, :3] = ~image_numpy[:, :, :3]
		
		# Remove pixels that do not pass the filter
		image_numpy[:, :, 3] *= selected_mask
		masked_image = Image.fromarray(image_numpy)
		output_image = Image.alpha_composite(document.current.convert("RGBA"), masked_image).convert("RGB")

		mask_source = "background" if selection.use_background else "foreground"
		return ToolResult(
			image=output_image,
			message=f"Applied {mask_source}-brightness mask to {Path(filename).name}.",
			details={
				"Operation": "Masking composite",
				"Threshold": selection.threshold,
				"Binary output": selection.binary,
				"Retained pixels": "Bright" if selection.keep_bright else "Dark",
				"Mask source": mask_source.title(),
				"Image inverted": selection.invert_image,
				"Foreground": Path(filename).name,
				"Output mode": output_image.mode,
			},
		)