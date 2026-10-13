"""Apply a mean convolution with a selectable kernel shape."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image
from scipy import ndimage

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utils import add_dialog_buttons, create_dialog, wait_for_dialog


@dataclass(frozen=True)
class ConvolutionSettings:
	size: int
	direction: str


def choose_convolution(parent: tk.Misc, *, title: str) -> ConvolutionSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: ConvolutionSettings | None = None
	size_value = tk.StringVar(master=dialog, value="15")
	direction_value = tk.StringVar(master=dialog, value="both")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		value = size_value.get()
		if not value.isdecimal() or int(value) < 1:
			error_text.set("Enter a kernel size greater than 0.")
			return
		selected = ConvolutionSettings(int(value), direction_value.get())
		dialog.destroy()

	ttk.Label(dialog, text="Kernel size:").pack(anchor="w", padx=16, pady=(14, 4))
	ttk.Spinbox(dialog, from_=1, to=255, textvariable=size_value, width=7).pack(
		anchor="w", padx=16, pady=(0, 8)
	)
	ttk.Label(dialog, text="Direction:").pack(anchor="w", padx=16)
	directions = ttk.Frame(dialog)
	directions.pack(anchor="w", padx=12, pady=(0, 6))
	for direction in ("vertical", "horizontal", "both"):
		ttk.Radiobutton(
			directions,
			text=direction.title(),
			variable=direction_value,
			value=direction,
		).pack(side="left", padx=4)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16)
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


class ConvolutionTool(ForensicsTool):
	tool_id = "convolution"
	title = "Convolution"
	category = "Set 3"
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
