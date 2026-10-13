"""Sharpen images with an unsharp mask."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .utils import add_dialog_buttons, add_numeric_field, create_dialog, wait_for_dialog


@dataclass(frozen=True)
class SharpeningSettings:
	radius: float
	amount: float


def choose_sharpening(parent: tk.Misc, *, title: str) -> SharpeningSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: SharpeningSettings | None = None
	radius_value = tk.StringVar(master=dialog, value="2.0")
	amount_value = tk.StringVar(master=dialog, value="1.5")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			radius = float(radius_value.get())
			amount = float(amount_value.get())
		except ValueError:
			error_text.set("Enter valid numeric values.")
			return
		if radius <= 0 or amount < 0:
			error_text.set("Use a positive radius and an amount of 0 or more.")
			return
		selected = SharpeningSettings(radius, amount)
		dialog.destroy()

	for label, value in (("Radius:", radius_value), ("Amount:", amount_value)):
		add_numeric_field(dialog, label, value)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


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
