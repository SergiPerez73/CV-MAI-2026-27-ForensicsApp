"""Apply common smoothing and edge-detection filters."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .constants import FILTER_NAMES
from .utils import add_dialog_buttons, create_dialog, wait_for_dialog


@dataclass(frozen=True)
class FilterSettings:
	name: str
	sigma: float


def choose_filter(parent: tk.Misc, *, title: str) -> FilterSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: FilterSettings | None = None
	filter_value = tk.StringVar(master=dialog, value="gaussian")
	sigma_value = tk.StringVar(master=dialog, value="2.0")
	error_text = tk.StringVar(master=dialog)

	def update_sigma_state(_event: tk.Event | None = None) -> None:
		sigma_control.configure(state="normal" if filter_value.get() == "gaussian" else "disabled")

	def apply_settings() -> None:
		nonlocal selected
		try:
			sigma = float(sigma_value.get())
		except ValueError:
			error_text.set("Enter a sigma between 0.1 and 20.")
			return
		if not 0.1 <= sigma <= 20:
			error_text.set("Enter a sigma between 0.1 and 20.")
			return
		selected = FilterSettings(filter_value.get(), sigma)
		dialog.destroy()

	ttk.Label(dialog, text="Filter:").pack(anchor="w", padx=16, pady=(14, 4))
	filter_control = ttk.Combobox(
		dialog,
		textvariable=filter_value,
		values=("gaussian", "median", "sobel"),
		state="readonly",
		width=18,
	)
	filter_control.pack(anchor="w", padx=16, pady=(0, 8))
	filter_control.bind("<<ComboboxSelected>>", update_sigma_state)
	ttk.Label(dialog, text="Gaussian sigma:").pack(anchor="w", padx=16)
	sigma_control = ttk.Spinbox(
		dialog,
		from_=0.1,
		to=20,
		increment=0.1,
		textvariable=sigma_value,
		width=7,
	)
	sigma_control.pack(anchor="w", padx=16, pady=(0, 6))
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16)
	add_dialog_buttons(dialog, apply_settings)
	update_sigma_state()
	wait_for_dialog(parent, dialog)
	return selected


class FiltersTool(ForensicsTool):
	tool_id = "filters"
	title = "Filters"
	category = "Set 3"
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
