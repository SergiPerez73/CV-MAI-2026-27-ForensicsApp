"""Apply contrast transformations to an image."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image
from skimage import exposure
from skimage.color import lab2rgb, rgb2lab
from skimage.util import img_as_float, img_as_ubyte

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..constants import CONTRAST_OPTIONS
from ..utils import choose_option


class ContrastEnhancementTool(ForensicsTool):
	tool_id = "contrast_enhancement"
	title = "Contrast enhancement"
	category = "Set 2"
	description = "Apply low-contrast rescaling, contrast stretching, histogram equalization, or CLAHE."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		method = choose_option(
			parent,
			title=self.title,
			prompt="Choose a contrast method:",
			options=CONTRAST_OPTIONS,
		)
		if method is None:
			return None

		image = document.current
		image_numpy = np.asarray(image)
		lab = None
		if image_numpy.ndim == 2:
			intensity = img_as_float(image_numpy)
		else:
			rgb = img_as_float(np.asarray(image.convert("RGB")))
			lab = rgb2lab(rgb)
			intensity = lab[:, :, 0] / 100

		low_contrast = exposure.rescale_intensity(
			intensity,
			in_range=(0, 1),
			out_range=(0.3, 0.7),
		)

		if method == "low_contrast":
			result = low_contrast
		elif method == "stretch":
			lower, upper = np.percentile(low_contrast, (2, 98))
			result = exposure.rescale_intensity(
				low_contrast,
				in_range=(lower, upper),
				out_range=(0, 1),
			)
		elif method == "equalize":
			result = exposure.equalize_hist(low_contrast)
		elif method == "clahe":
			result = exposure.equalize_adapthist(low_contrast, clip_limit=0.03)

        # If the image is in lab color space, convert it back to RGB.
		if lab is not None:
			lab[:, :, 0] = result * 100
			result = np.clip(lab2rgb(lab), 0, 1)

		output_image = Image.fromarray(img_as_ubyte(result))
		label = CONTRAST_OPTIONS[method]
		return ToolResult(
			image=output_image,
			message=f"Applied {label.lower()}.",
			details={
				"Operation": label,
				"Output mode": output_image.mode,
			},
		)