"""Create binary or pixel-preserving threshold masks."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageOps

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .constants import (
	DEFAULT_BINARY,
	DEFAULT_INVERT_IMAGE,
	DEFAULT_KEEP_BRIGHT,
	DEFAULT_THRESHOLD,
	MASK_OPTIONS,
	MAX_THRESHOLD,
	MIN_THRESHOLD,
	OPEN_TYPES,
)
from .utils import add_dialog_buttons, create_dialog, wait_for_dialog


@dataclass(frozen=True)
class MaskingSettings:
	binary: bool
	keep_bright: bool
	invert_image: bool
	threshold: int
	use_background: bool = False


def choose_masking(
	parent: tk.Misc,
	*,
	title: str,
	prompt: str,
	default_threshold: int = DEFAULT_THRESHOLD,
) -> MaskingSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: MaskingSettings | None = None
	binary_value = tk.BooleanVar(master=dialog, value=DEFAULT_BINARY)
	keep_bright_value = tk.BooleanVar(master=dialog, value=DEFAULT_KEEP_BRIGHT)
	invert_image_value = tk.BooleanVar(master=dialog, value=DEFAULT_INVERT_IMAGE)
	use_background_value = tk.BooleanVar(master=dialog, value=False)
	threshold_value = tk.DoubleVar(master=dialog, value=default_threshold)
	threshold_text = tk.StringVar(master=dialog, value=str(default_threshold))
	error_text = tk.StringVar(master=dialog)
	updating = False

	def validate_threshold(value: str) -> bool:
		if not value:
			return True
		return value.isdecimal() and MIN_THRESHOLD <= int(value) <= MAX_THRESHOLD

	def update_text(value: str) -> None:
		nonlocal updating
		if not updating:
			updating = True
			threshold_text.set(str(round(float(value))))
			updating = False

	def update_scale(*_args: str) -> None:
		nonlocal updating
		value = threshold_text.get()
		if not updating and value.isdecimal():
			updating = True
			threshold_value.set(int(value))
			updating = False

	def apply_settings() -> None:
		nonlocal selected
		value = threshold_text.get()
		if not value:
			error_text.set("Enter a threshold from 0 to 255.")
			return
		selected = MaskingSettings(
			binary=binary_value.get(),
			keep_bright=keep_bright_value.get(),
			invert_image=invert_image_value.get(),
			threshold=int(value),
			use_background=use_background_value.get(),
		)
		dialog.destroy()

	threshold_text.trace_add("write", update_scale)
	ttk.Label(dialog, text=prompt).pack(anchor="w", padx=16, pady=(14, 8))
	ttk.Checkbutton(dialog, text="Binary output", variable=binary_value).pack(
		anchor="w", padx=16, pady=(0, 6)
	)
	ttk.Label(dialog, text="Retain pixels:").pack(anchor="w", padx=16)
	retention = ttk.Frame(dialog)
	retention.pack(anchor="w", padx=12, pady=(0, 6))
	for keep_bright, label in ((True, MASK_OPTIONS["bright"]), (False, MASK_OPTIONS["dark"])):
		ttk.Radiobutton(
			retention,
			text=label,
			variable=keep_bright_value,
			value=keep_bright,
		).pack(side="left", padx=4)
	ttk.Label(dialog, text="Select pixels from:").pack(anchor="w", padx=16)
	mask_source = ttk.Frame(dialog)
	mask_source.pack(anchor="w", padx=12, pady=(0, 6))
	for use_background, label in ((False, "Image on top"), (True, "Background image")):
		ttk.Radiobutton(
			mask_source,
			text=label,
			variable=use_background_value,
			value=use_background,
		).pack(side="left", padx=4)
	ttk.Checkbutton(
		dialog,
		text="Invert image before masking",
		variable=invert_image_value,
	).pack(anchor="w", padx=16, pady=(0, 6))
	ttk.Label(dialog, text="Threshold (0-255):").pack(anchor="w", padx=16)
	controls = ttk.Frame(dialog)
	controls.pack(fill="x", padx=12, pady=(0, 4))
	ttk.Scale(
		controls,
		from_=MIN_THRESHOLD,
		to=MAX_THRESHOLD,
		variable=threshold_value,
		command=update_text,
	).pack(side="left", fill="x", expand=True, padx=(0, 8))
	validation = (dialog.register(validate_threshold), "%P")
	ttk.Spinbox(
		controls,
		from_=MIN_THRESHOLD,
		to=MAX_THRESHOLD,
		textvariable=threshold_text,
		validate="key",
		validatecommand=validation,
		width=5,
	).pack(side="left")
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16)
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


class MaskingTool(ForensicsTool):
	tool_id = "masking"
	title = "Masking"
	category = "Set 2"
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