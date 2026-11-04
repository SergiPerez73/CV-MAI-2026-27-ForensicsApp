"""Retrieve the closest training faces using a saved eigenface model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utils import add_dialog_buttons, add_numeric_field, create_dialog, figure_to_image, wait_for_dialog
from .eigenface_builder import DEFAULT_DATA
from .eigenfaces import EigenfaceModel, FaceVote, inverse_distance_vote, load_eigenface_model, nearest_faces, resolve_training_folder


@dataclass(frozen=True)
class FaceRecognitionSettings:
	count: int = 3


def choose_face_recognition(parent: tk.Misc, *, title: str) -> FaceRecognitionSettings | None:
	dialog = create_dialog(parent, title=title)
	selected: FaceRecognitionSettings | None = None
	count_value = tk.StringVar(master=dialog, value="3")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			count = int(count_value.get())
			if count < 1:
				raise ValueError
		except ValueError:
			error_text.set("Enter a positive whole number of similar faces.")
			return
		selected = FaceRecognitionSettings(count)
		dialog.destroy()

	add_numeric_field(dialog, "Similar faces to retrieve:", count_value)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	add_dialog_buttons(dialog, apply_settings)
	wait_for_dialog(parent, dialog)
	return selected


def render_face_matches(
	image: Image.Image, model: EigenfaceModel, indices: np.ndarray, distances: np.ndarray,
	vote: FaceVote | None = None,
) -> Image.Image:
	count = len(indices) + 1
	columns = min(4, count)
	rows = (count + columns - 1) // columns
	figure = Figure(figsize=(3 * columns, 3.5 * rows), dpi=100, layout="constrained")
	if vote is not None:
		figure.suptitle(f"Weighted vote: {vote.label} ({vote.scores[vote.label]:.1%})", fontsize=12)
	axes = figure.subplots(rows, columns, squeeze=False).flatten()
	height, width = model.image_shape
	axes[0].imshow(image.convert("L").resize((width, height), Image.Resampling.BILINEAR), cmap="gray", vmin=0, vmax=255)
	axes[0].set_title("Test image", fontsize=10)
	for rank, (index, distance) in enumerate(zip(indices, distances), start=1):
		path = model.paths[int(index)]
		if not path.is_file():
			raise FileNotFoundError(f"Training image not found: {path}. Keep the original dataset at its saved location.")
		with Image.open(path) as source:
			face = source.convert("L").resize((width, height), Image.Resampling.BILINEAR)
		axes[rank].imshow(face, cmap="gray", vmin=0, vmax=255)
		title = f"{rank}. {model.labels[index]}\nDistance: {float(distance):.3f}"
		if vote is not None:
			title += f"\nWeight: {vote.weights[rank - 1]:.1%}"
		axes[rank].set_title(title, fontsize=10, wrap=True)
	for axis in axes:
		axis.set_axis_off()
	return figure_to_image(figure)


class FaceRecognitionTool(ForensicsTool):
	tool_id = "face_recognition"
	title = "Face recognition"
	category = "Set 6"
	description = "Retrieve similar training faces and combine their labels with inverse-distance voting."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
		assert document.current is not None
		models = sorted(DEFAULT_DATA.glob("output_*/eigenfaces.npz"))
		model_name = filedialog.askopenfilename(
			parent=parent, title="Choose eigenface model", filetypes=[("Eigenface model", "*.npz")],
			initialdir=str(models[-1].parent if models else DEFAULT_DATA), initialfile="eigenfaces.npz",
		)
		if not model_name:
			return None
		model = load_eigenface_model(Path(model_name))
		train_name = filedialog.askdirectory(
			parent=parent, title="Choose training image folder",
			initialdir=str(DEFAULT_DATA / "train_data" / "faces"),
		)
		if not train_name:
			return None
		settings = choose_face_recognition(parent, title=self.title)
		if settings is None:
			return None
		model = resolve_training_folder(model, Path(train_name))
		image = document.current
		indices, distances = nearest_faces(image, model, count=settings.count)
		vote = inverse_distance_vote(model.labels[indices], distances)
		output_image = render_face_matches(image, model, indices, distances, vote)
		best_name = str(model.labels[indices[0]])
		details: dict[str, str | int | float] = {
			"Operation": "Eigenface recognition",
			"Model": model_name,
			"Test image": "Currently open image",
			"Training folder": train_name,
			"Nearest name": best_name,
			"Vote result": vote.label,
			"Vote score": vote.scores[vote.label],
			"Requested matches": settings.count,
			"Retrieved matches": len(indices),
			"Weight sum": float(vote.weights.sum()),
			"Components (K)": len(model.eigenfaces),
			"Training images": len(model.labels),
			"Distance metric": "Euclidean distance in PCA space",
			"Output mode": output_image.mode,
		}
		for label, score in sorted(vote.scores.items(), key=lambda item: item[1], reverse=True):
			details[f"Vote: {label}"] = score
		for rank, (index, distance) in enumerate(zip(indices, distances), start=1):
			details[f"Match {rank}"] = str(model.labels[index])
			details[f"Distance {rank}"] = float(distance)
			details[f"Weight {rank}"] = float(vote.weights[rank - 1])
			details[f"Image {rank}"] = str(model.paths[int(index)])
		return ToolResult(
			image=output_image,
			message=f"Weighted vote: {vote.label} ({vote.scores[vote.label]:.1%}). Showing {len(indices)} closest images.",
			details=details,
		)