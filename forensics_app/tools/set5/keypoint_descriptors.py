"""Extract and visualize configurable ORB or SIFT keypoints."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from .utils import DescriptorSettings, extract_features, render_keypoints
from ..utils import add_dialog_buttons, add_numeric_field, create_dialog, wait_for_dialog


def choose_descriptor_settings(parent: tk.Misc, *, title: str) -> DescriptorSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: DescriptorSettings | None = None
	algorithm_value = tk.StringVar(master=dialog, value="orb")
	keypoints_value = tk.StringVar(master=dialog, value="200")
	octaves_value = tk.StringVar(master=dialog, value="3")
	scales_value = tk.StringVar(master=dialog, value="3")
	sigma_value = tk.StringVar(master=dialog, value="1.6")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			if algorithm_value.get() == "orb":
				settings = DescriptorSettings("orb", int(keypoints_value.get()), 3, 3, 1.6)
			else:
				settings = DescriptorSettings("sift", 200, int(octaves_value.get()), int(scales_value.get()), float(sigma_value.get()))
		except ValueError:
			error_text.set("Enter valid numeric values.")
			return
		if settings.n_keypoints < 1 or settings.n_octaves < 1 or settings.n_scales < 1 or settings.sigma_min <= 0:
			error_text.set("Use positive values.")
			return
		selected = settings
		dialog.destroy()

	def update_fields(*_args: object) -> None:
		orb_control.configure(state="normal" if algorithm_value.get() == "orb" else "disabled")
		sift_state = "normal" if algorithm_value.get() == "sift" else "disabled"
		for control in sift_controls:
			control.configure(state=sift_state)

	ttk.Label(dialog, text="Descriptor:").pack(anchor="w", padx=16, pady=(12, 0))
	ttk.Radiobutton(dialog, text="ORB", variable=algorithm_value, value="orb").pack(anchor="w", padx=16)
	ttk.Radiobutton(dialog, text="SIFT", variable=algorithm_value, value="sift").pack(anchor="w", padx=16)
	orb_control = add_numeric_field(dialog, "ORB keypoints:", keypoints_value)
	sift_controls = (
		add_numeric_field(dialog, "SIFT octaves:", octaves_value),
		add_numeric_field(dialog, "SIFT scales per octave:", scales_value),
		add_numeric_field(dialog, "SIFT minimum sigma:", sigma_value),
	)
	algorithm_value.trace_add("write", update_fields)
	update_fields()
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


class KeypointDescriptorsTool(ForensicsTool):
	tool_id = "keypoint_descriptors"
	title = "Descriptor keypoints"
	category = "Set 5"
	description = "Extract and visualize configurable ORB or SIFT keypoints."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		settings = choose_descriptor_settings(parent, title=self.title)
		if settings is None:
			return None

		keypoints, descriptors = extract_features(document.current, settings)
		output_image = render_keypoints(document.current, keypoints)
		algorithm = settings.algorithm.upper()
		details: dict[str, object] = {
			"Operation": f"{algorithm} descriptor extraction",
			"Keypoints": len(keypoints),
			"Descriptors": len(descriptors),
			"Output mode": output_image.mode,
		}
		if settings.algorithm == "orb":
			details["ORB keypoints"] = settings.n_keypoints
		else:
			details.update({
				"SIFT octaves": settings.n_octaves,
				"SIFT scales per octave": settings.n_scales,
				"SIFT minimum sigma": settings.sigma_min,
			})
		return ToolResult(
			image=output_image,
			message=f"Extracted {len(keypoints)} {algorithm} keypoints.",
			details=details,
		)