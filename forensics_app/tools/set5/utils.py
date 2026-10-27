"""Shared feature extraction, matching, and visualization helpers for Set 5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from skimage import color, feature

from ..utils import figure_to_image


@dataclass(frozen=True)
class DescriptorSettings:
	algorithm: str
	n_keypoints: int
	n_octaves: int
	n_scales: int
	sigma_min: float


@dataclass(frozen=True)
class MatchingSettings:
	cross_check: bool
	max_ratio: float


def extract_features(image: Image.Image, settings: DescriptorSettings) -> tuple[np.ndarray, np.ndarray]:
	"""Extract configured keypoints and descriptors from an RGB or grayscale image."""
	if settings.algorithm == "orb":
		descriptor = feature.ORB(n_keypoints=settings.n_keypoints)
	else:
		descriptor = feature.SIFT(
			n_octaves=settings.n_octaves,
			n_scales=settings.n_scales,
			sigma_min=settings.sigma_min,
		)
	array = np.array(image)
	descriptor.detect_and_extract(array if array.ndim == 2 else color.rgb2gray(array[..., :3]))
	return descriptor.keypoints, descriptor.descriptors


def match_features(
	reference_descriptors: np.ndarray,
	candidate_descriptors: np.ndarray,
	descriptor_settings: DescriptorSettings,
	matching_settings: MatchingSettings,
) -> np.ndarray:
	"""Match two descriptor sets using the appropriate metric for their algorithm."""
	if not len(reference_descriptors) or not len(candidate_descriptors):
		return np.empty((0, 2), dtype=int)
	metric = "hamming" if descriptor_settings.algorithm == "orb" else "euclidean"
	return feature.match_descriptors(
		reference_descriptors,
		candidate_descriptors,
		metric=metric,
		cross_check=matching_settings.cross_check,
		max_ratio=matching_settings.max_ratio,
	)


def render_keypoints(image: Image.Image, keypoints: np.ndarray) -> Image.Image:
	"""Draw detected keypoints as red hollow circles over an image."""
	array = np.array(image.convert("RGB"))
	figure = Figure(figsize=(array.shape[1] / 100, array.shape[0] / 100), dpi=100)
	axis = figure.add_axes((0, 0, 1, 1))
	axis.imshow(array)
	axis.plot(keypoints[:, 1], keypoints[:, 0], "ro", markersize=5, markerfacecolor="none")
	axis.axis("off")
	return figure_to_image(figure)


def render_matches(
	reference: Image.Image,
	candidate: Image.Image,
	reference_keypoints: np.ndarray,
	candidate_keypoints: np.ndarray,
	matches: np.ndarray,
) -> Image.Image:
	"""Draw descriptor correspondences between two images."""
	reference_array = np.array(reference.convert("RGB"))
	candidate_array = np.array(candidate.convert("RGB"))
	height = max(reference_array.shape[0], candidate_array.shape[0])
	width = reference_array.shape[1] + candidate_array.shape[1]
	combined = np.zeros((height, width, 3), dtype=np.uint8)
	combined[:reference_array.shape[0], :reference_array.shape[1]] = reference_array
	combined[:candidate_array.shape[0], reference_array.shape[1]:] = candidate_array
	figure = Figure(figsize=(width / 100, height / 100), dpi=100)
	axis = figure.add_axes((0, 0, 1, 1))
	axis.imshow(combined)
	for reference_index, candidate_index in matches:
		reference_y, reference_x = reference_keypoints[reference_index]
		candidate_y, candidate_x = candidate_keypoints[candidate_index]
		candidate_x += reference_array.shape[1]
		axis.plot((reference_x, candidate_x), (reference_y, candidate_y), color="cyan", linewidth=0.8)
		axis.plot(reference_x, reference_y, "ro", markersize=4, markerfacecolor="none")
		axis.plot(candidate_x, candidate_y, "ro", markersize=4, markerfacecolor="none")
	axis.axis("off")
	return figure_to_image(figure)
