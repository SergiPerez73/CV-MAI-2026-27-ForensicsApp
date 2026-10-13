"""Shared values used by image-processing tools."""

CHANNELS = {"Red": 0, "Green": 1, "Blue": 2}
OPEN_TYPES = [
	("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
	("All files", "*.*"),
]

MIN_THRESHOLD = 0
MAX_THRESHOLD = 255
DEFAULT_THRESHOLD = 135
DEFAULT_BINARY = False
DEFAULT_KEEP_BRIGHT = True
DEFAULT_INVERT_IMAGE = False
MASK_OPTIONS = {
	"bright": "Select bright pixels",
	"dark": "Select dark pixels",
}
FILTER_NAMES = {
	"gaussian": "Gaussian smoothing",
	"median": "Median denoising",
	"sobel": "Sobel edge detection",
}
CONTRAST_OPTIONS = {
	"low_contrast": "Create low-contrast image",
	"stretch": "Contrast stretching",
	"equalize": "Histogram equalization",
	"clahe": "Adaptive histogram equalization (CLAHE)",
}