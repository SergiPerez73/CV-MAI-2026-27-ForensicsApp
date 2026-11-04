"""Dataset preparation, PCA fitting, and persistence for eigenfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class EigenfaceModel:
	mean: np.ndarray
	eigenfaces: np.ndarray
	coefficients: np.ndarray
	labels: np.ndarray
	paths: tuple[Path, ...]
	image_shape: tuple[int, int]


@dataclass(frozen=True)
class FaceVote:
	label: str
	weights: np.ndarray
	scores: dict[str, float]


def inverse_distance_vote(labels: np.ndarray, distances: np.ndarray) -> FaceVote:
	distances = np.array(distances, dtype=np.float64)
	if labels.ndim != 1 or distances.ndim != 1 or len(labels) != len(distances) or len(labels) == 0:
		raise ValueError("Provide one label and distance for each retrieved face.")
	if not np.isfinite(distances).all() or np.any(distances < 0):
		raise ValueError("Distances must be finite and nonnegative.")
	exact_matches = distances == 0
	if exact_matches.any():
		weights = exact_matches.astype(np.float64) / exact_matches.sum()
	else:
		# Scaling reciprocals by the smallest distance avoids overflow without
		# changing their normalized proportions: weights still sum to one.
		weights = distances.min() / distances
		weights /= weights.sum()
	scores: dict[str, float] = {}
	for label, weight in zip(labels, weights):
		name = str(label)
		scores[name] = scores.get(name, 0.0) + float(weight)
	# Input order is nearest-first, so equal label totals favor the nearest label.
	return FaceVote(max(scores, key=lambda name: scores[name]), weights, scores)


def load_eigenface_model(path: Path) -> EigenfaceModel:
	fields = ("mean", "eigenfaces", "coefficients", "train_labels", "train_paths", "image_shape")
	with np.load(path, allow_pickle=False) as archive:
		if not set(fields).issubset(archive.files):
			raise ValueError("The selected file is missing required eigenface model arrays.")
		mean, eigenfaces, coefficients, labels, paths, shape = (np.array(archive[field]) for field in fields)
	if shape.shape != (2,) or not np.issubdtype(shape.dtype, np.integer) or np.any(shape <= 0):
		raise ValueError("The model must contain a positive grayscale image height and width.")
	height, width = (int(value) for value in shape)
	if (
		mean.shape != (height * width,)
		or eigenfaces.ndim != 2 or eigenfaces.shape[1] != len(mean) or len(eigenfaces) == 0
		or coefficients.ndim != 2 or coefficients.shape[1] != len(eigenfaces) or len(coefficients) == 0
		or labels.shape != (len(coefficients),) or paths.shape != (len(coefficients),)
	):
		raise ValueError("The saved mean, eigenfaces, coefficients, labels, and paths have incompatible sizes.")
	for array in (mean, eigenfaces, coefficients):
		if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array) or not np.isfinite(array).all():
			raise ValueError("Model values must be finite real numbers.")
	if labels.dtype.kind not in "US" or paths.dtype.kind not in "US" or np.any(labels == "") or np.any(paths == ""):
		raise ValueError("The model must contain a name and image path for each training face.")
	image_paths = tuple(
		Path(str(value)) if Path(str(value)).is_absolute() else path.parent / str(value) for value in paths
	)
	return EigenfaceModel(mean, eigenfaces, coefficients, labels.astype(str), image_paths, (height, width))


def resolve_training_folder(model: EigenfaceModel, folder: Path) -> EigenfaceModel:
	face_folder = folder / "faces" if (folder / "faces").is_dir() else folder
	files: dict[str, list[Path]] = {}
	for path in sorted(face_folder.rglob("*")):
		if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
			if "non_faces" not in path.relative_to(face_folder).parts[:-1]:
				files.setdefault(path.name, []).append(path.resolve())
	paths = []
	for saved_path, label in zip(model.paths, model.labels):
		candidates = files.get(saved_path.name, [])
		if len(candidates) > 1:
			candidates = [path for path in candidates if path.parent.name == str(label)]
		if len(candidates) != 1:
			raise ValueError(
				f"Cannot uniquely locate training image {saved_path.name} ({label}) in the selected folder. "
				"Choose the training dataset used to build this model."
			)
		paths.append(candidates[0])
	return replace(model, paths=tuple(paths))


def nearest_faces(image: Image.Image, model: EigenfaceModel, count: int = 3) -> tuple[np.ndarray, np.ndarray]:
	if count < 1:
		raise ValueError("Request at least one matching face.")
	height, width = model.image_shape
	face = image.convert("L").resize((width, height), Image.Resampling.BILINEAR)
	pixels = np.array(face, dtype=np.float32).reshape(-1) / 255.0
	# Project with the saved training mean and basis; do not fit PCA to the test face.
	coefficients = model.eigenfaces @ (pixels - model.mean)
	distances = np.linalg.norm(model.coefficients - coefficients, axis=1)
	indices = np.argsort(distances, kind="stable")[:count]
	return indices, distances[indices]


def load_face_folder(folder: Path, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	images = []
	labels = []
	paths = []
	# The tutorial splits contain faces/ and non_faces/. Only faces belong in PCA.
	face_folder = folder / "faces" if (folder / "faces").is_dir() else folder
	for path in sorted(face_folder.rglob("*")):
		if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
			continue
		relative_path = path.relative_to(face_folder)
		if "non_faces" in relative_path.parts[:-1]:
			continue
		with Image.open(path) as source:
			image = source.convert("L").resize(size, Image.Resampling.BILINEAR)
		images.append(np.array(image, dtype=np.float32) / 255.0)
		# Person subfolders supply names; flat face folders only supply sample IDs.
		labels.append(relative_path.parts[0] if len(relative_path.parts) > 1 else path.stem)
		paths.append(str(path.resolve()))
	if not images:
		raise ValueError("No face images found in the selected folder.")
	return np.array(images), np.array(labels, dtype=str), np.array(paths, dtype=str)


def fit_eigenfaces(train_images: np.ndarray, components: int) -> PCA:
	if train_images.ndim != 3 or len(train_images) < 2:
		raise ValueError("PCA needs at least two grayscale training images of the same size.")
	faces = train_images.reshape(len(train_images), -1)
	maximum = min(len(faces) - 1, faces.shape[1])
	if not 1 <= components <= maximum:
		raise ValueError(f"K must be between 1 and {maximum} for this training set.")
	if not np.isfinite(faces).all() or not np.any(faces != faces[0]):
		raise ValueError("Training images must be finite and must not all be identical.")
	# sklearn centers each face by the training mean. No whitening: coefficients
	# are exactly the projections (face - mean) @ eigenfaces.T used in the PDF.
	return PCA(n_components=components, svd_solver="randomized", whiten=False, random_state=42).fit(faces)


def save_eigenface_dataset(
	folder: Path,
	train_images: np.ndarray,
	train_labels: np.ndarray,
	train_paths: np.ndarray,
	test_labels: np.ndarray,
	test_paths: np.ndarray,
	pca: PCA,
) -> Path:
	if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
		raise ValueError("Choose a new or empty output folder to avoid overwriting a model.")
	folder.mkdir(parents=True, exist_ok=True)
	manifest = []
	# Reference existing files instead of copying images or creating a new split.
	for split_name, labels, paths in (("train", train_labels, train_paths), ("test", test_labels, test_paths)):
		for label, path in zip(labels, paths):
			manifest.append({"path": str(path), "name": str(label), "split": split_name})
	with (folder / "labels.json").open("w", encoding="utf-8") as stream:
		json.dump(manifest, stream, indent=2, ensure_ascii=True)
	train_faces = train_images.reshape(len(train_images), -1)
	model_path = folder / "eigenfaces.npz"
	np.savez_compressed(
		model_path,
		mean=pca.mean_,
		eigenfaces=pca.components_,
		coefficients=pca.transform(train_faces),
		train_labels=np.array(train_labels, dtype=str),
		train_paths=np.array(train_paths, dtype=str),
		test_labels=np.array(test_labels, dtype=str),
		test_paths=np.array(test_paths, dtype=str),
		image_shape=np.array(train_images.shape[1:], dtype=np.int64),
		explained_variance_ratio=pca.explained_variance_ratio_,
	)
	return model_path