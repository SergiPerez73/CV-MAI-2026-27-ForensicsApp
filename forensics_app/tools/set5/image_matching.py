"""Match configurable local descriptors against a second image."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..constants import OPEN_TYPES
from .utils import MatchingSettings, extract_features, match_features, render_matches
from .keypoint_descriptors import choose_descriptor_settings
from ..utils import add_dialog_buttons, add_numeric_field, create_dialog, wait_for_dialog


def choose_matching_settings(parent: tk.Misc, *, title: str) -> MatchingSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: MatchingSettings | None = None
	mode_value = tk.StringVar(master=dialog, value="cross_check")
	ratio_value = tk.StringVar(master=dialog, value="0.8")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		if mode_value.get() == "cross_check":
			selected = MatchingSettings(True, 1.0)
			dialog.destroy()
			return
		try:
			ratio = float(ratio_value.get())
		except ValueError:
			error_text.set("Enter a numeric ratio.")
			return
		if not 0 < ratio <= 1:
			error_text.set("Use a ratio greater than 0 and at most 1.")
			return
		selected = MatchingSettings(False, ratio)
		dialog.destroy()

	def update_fields(*_args: object) -> None:
		ratio_control.configure(state="disabled" if mode_value.get() == "cross_check" else "normal")

	ttk.Label(dialog, text="Matching method:").pack(anchor="w", padx=16, pady=(12, 0))
	ttk.Radiobutton(dialog, text="Cross-check", variable=mode_value, value="cross_check").pack(anchor="w", padx=16)
	ttk.Radiobutton(dialog, text="Ratio test", variable=mode_value, value="ratio").pack(anchor="w", padx=16)
	ratio_control = add_numeric_field(dialog, "Maximum ratio:", ratio_value)
	mode_value.trace_add("write", update_fields)
	update_fields()
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


class ImageMatchingTool(ForensicsTool):
	tool_id = "image_matching"
	title = "Match second image"
	category = "Set 5"
	description = "Match configurable ORB or SIFT descriptors with a second image."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		filename = filedialog.askopenfilename(title="Choose second image", filetypes=OPEN_TYPES)
		if not filename:
			return None
		descriptor_settings = choose_descriptor_settings(parent, title=self.title)
		if descriptor_settings is None:
			return None
		matching_settings = choose_matching_settings(parent, title=self.title)
		if matching_settings is None:
			return None

		with Image.open(filename) as source:
			candidate = source.copy()
		reference_keypoints, reference_descriptors = extract_features(document.current, descriptor_settings)
		candidate_keypoints, candidate_descriptors = extract_features(candidate, descriptor_settings)
		matches = match_features(reference_descriptors, candidate_descriptors, descriptor_settings, matching_settings)
		output_image = render_matches(
			document.current,
			candidate,
			reference_keypoints,
			candidate_keypoints,
			matches,
		)

		return ToolResult(
			image=output_image,
			message=f"Found {len(matches)} matches with {Path(filename).name}.",
			details={
				"Operation": f"{descriptor_settings.algorithm.upper()} descriptor matching",
				"Second image": Path(filename).name,
				"Matches": len(matches),
				"Matching method": "Cross-check" if matching_settings.cross_check else "Ratio test",
				"Output mode": output_image.mode,
			},
		)