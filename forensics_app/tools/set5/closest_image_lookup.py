"""Find the image in a folder with the most descriptor matches."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from .utils import extract_features, match_features, render_matches
from .image_matching import choose_matching_settings
from .keypoint_descriptors import choose_descriptor_settings


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class ClosestImageLookupTool(ForensicsTool):
	tool_id = "closest_image_lookup"
	title = "Lookup closest image"
	category = "Set 5"
	description = "Find the folder image with the largest number of descriptor matches."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		folder_name = filedialog.askdirectory(title="Choose image folder")
		if not folder_name:
			return None
		descriptor_settings = choose_descriptor_settings(parent, title=self.title)
		if descriptor_settings is None:
			return None
		matching_settings = choose_matching_settings(parent, title=self.title)
		if matching_settings is None:
			return None

		folder = Path(folder_name)
		reference_keypoints, reference_descriptors = extract_features(document.current, descriptor_settings)
		source_path = document.path.resolve() if document.path else None
		best_path: Path | None = None
		best_image: Image.Image | None = None
		best_keypoints = None
		best_matches = None
		examined = 0
		skipped = 0

		for path in sorted(folder.iterdir()):
			if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES or path.resolve() == source_path:
				continue
			try:
				with Image.open(path) as source:
					candidate = source.copy()
			except OSError:
				skipped += 1
				continue
			examined += 1
			candidate_keypoints, candidate_descriptors = extract_features(candidate, descriptor_settings)
			matches = match_features(reference_descriptors, candidate_descriptors, descriptor_settings, matching_settings)
			if best_matches is None or len(matches) > len(best_matches):
				best_path = path
				best_image = candidate
				best_keypoints = candidate_keypoints
				best_matches = matches

		if best_path is None or best_image is None or best_keypoints is None or best_matches is None:
			raise ValueError("The selected folder contains no usable image files.")
		output_image = render_matches(
			document.current,
			best_image,
			reference_keypoints,
			best_keypoints,
			best_matches,
		)

		return ToolResult(
			image=output_image,
			message=f"Closest image: {best_path.name} ({len(best_matches)} matches).",
			details={
				"Operation": f"{descriptor_settings.algorithm.upper()} folder lookup",
				"Folder": folder.name,
				"Closest image": best_path.name,
				"Matches": len(best_matches),
				"Candidates examined": examined,
				"Unreadable files": skipped,
				"Output mode": output_image.mode,
			},
		)