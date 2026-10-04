"""Shared values used by image-processing tools."""

CHANNELS = {"Red": 0, "Green": 1, "Blue": 2}

MIN_THRESHOLD = 0
MAX_THRESHOLD = 255
DEFAULT_THRESHOLD = 135
DEFAULT_BINARY = False
DEFAULT_KEEP_BRIGHT = True
DEFAULT_INVERT_IMAGE = False
MASK_OPTIONS = {
	"bright": "Keep bright pixels",
	"dark": "Keep dark pixels",
}
MASK_OPERATIONS = {
	"binary": "Binary mask",
	"bright": "Bright-pixel mask",
	"dark": "Dark-pixel mask",
}
MASK_MESSAGES = {
	"binary": "Created a binary mask for {region} pixels at threshold {threshold}.",
	"bright": "Kept pixels brighter than {threshold}.",
	"dark": "Kept pixels at or below {threshold}.",
}
CONTRAST_OPTIONS = {
	"low_contrast": "Create low-contrast image",
	"stretch": "Contrast stretching",
	"equalize": "Histogram equalization",
	"clahe": "Adaptive histogram equalization (CLAHE)",
}