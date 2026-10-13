from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

from .constants import (
	CHANNELS,
	DEFAULT_BINARY,
	DEFAULT_INVERT_IMAGE,
	DEFAULT_KEEP_BRIGHT,
	DEFAULT_THRESHOLD,
	MASK_OPTIONS,
	MAX_THRESHOLD,
	MIN_THRESHOLD,
)


@dataclass(frozen=True)
class MaskingSettings:
	binary: bool
	keep_bright: bool
	invert_image: bool
	threshold: int
	use_background: bool = False


@dataclass(frozen=True)
class ConvolutionSettings:
	size: int
	direction: str


@dataclass(frozen=True)
class FilterSettings:
	name: str
	sigma: float


@dataclass(frozen=True)
class CannySettings:
	sigma: float
	low_threshold: float
	high_threshold: float


@dataclass(frozen=True)
class EdgeVisualizationSettings:
	sigma: float
	low_threshold: float
	high_threshold: float
	line_width: float


@dataclass(frozen=True)
class SharpeningSettings:
	radius: float
	amount: float


def choose_option(
	parent: tk.Misc,
	*,
	title: str,
	prompt: str,
	options: dict[str, str],
	disabled_options: set[str] | None = None,
) -> str | None:
	dialog = tk.Toplevel(parent)
	dialog.title(title)
	dialog.transient(parent)
	dialog.resizable(False, False)

	selected_channel: str | None = None
	disabled = disabled_options or set()

	def select_channel(channel: str) -> None:
		nonlocal selected_channel
		selected_channel = channel
		dialog.destroy()

	ttk.Label(dialog, text=prompt).pack(padx=16, pady=(14, 8))
	buttons = ttk.Frame(dialog)
	buttons.pack(padx=12, pady=(0, 12))
	for value, label in options.items():
		ttk.Button(
			buttons,
			text=label,
			command=lambda choice=value: select_channel(choice),
			state="disabled" if value in disabled else "normal",
		).pack(side="left", padx=4)

	dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
	dialog.bind("<Escape>", lambda _event: dialog.destroy())
	dialog.update_idletasks()
	owner = parent.winfo_toplevel()
	x = owner.winfo_rootx() + (owner.winfo_width() - dialog.winfo_width()) // 2
	y = owner.winfo_rooty() + (owner.winfo_height() - dialog.winfo_height()) // 2
	dialog.geometry(f"+{x}+{y}")
	dialog.grab_set()
	parent.wait_window(dialog)
	return selected_channel


def choose_channel(
	parent: tk.Misc,
	*,
	title: str = "Choose color channel",
	prompt: str = "Choose a channel:",
	excluded_channels: set[str] | None = None,
) -> str | None:
	return choose_option(
		parent,
		title=title,
		prompt=prompt,
		options={channel: channel for channel in CHANNELS},
		disabled_options=excluded_channels,
	)


def choose_convolution(
	parent: tk.Misc,
	*,
	title: str,
) -> ConvolutionSettings | None:
	dialog = tk.Toplevel(parent)
	dialog.title(title)
	dialog.transient(parent)
	dialog.resizable(False, False)

	selected: ConvolutionSettings | None = None
	size_value = tk.StringVar(master=dialog, value="15")
	direction_value = tk.StringVar(master=dialog, value="both")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		value = size_value.get()
		if not value.isdecimal() or int(value) < 1:
			error_text.set("Enter a kernel size greater than 0.")
			return
		selected = ConvolutionSettings(int(value), direction_value.get())
		dialog.destroy()

	ttk.Label(dialog, text="Kernel size:").pack(anchor="w", padx=16, pady=(14, 4))
	ttk.Spinbox(dialog, from_=1, to=255, textvariable=size_value, width=7).pack(
		anchor="w", padx=16, pady=(0, 8)
	)
	ttk.Label(dialog, text="Direction:").pack(anchor="w", padx=16)
	directions = ttk.Frame(dialog)
	directions.pack(anchor="w", padx=12, pady=(0, 6))
	for direction in ("vertical", "horizontal", "both"):
		ttk.Radiobutton(
			directions,
			text=direction.title(),
			variable=direction_value,
			value=direction,
		).pack(side="left", padx=4)
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16)
	buttons = ttk.Frame(dialog)
	buttons.pack(padx=12, pady=(6, 12))
	ttk.Button(buttons, text="Apply", command=apply_settings).pack(side="left", padx=4)
	tk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)

	dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
	dialog.bind("<Escape>", lambda _event: dialog.destroy())
	dialog.update_idletasks()
	owner = parent.winfo_toplevel()
	x = owner.winfo_rootx() + (owner.winfo_width() - dialog.winfo_width()) // 2
	y = owner.winfo_rooty() + (owner.winfo_height() - dialog.winfo_height()) // 2
	dialog.geometry(f"+{x}+{y}")
	dialog.grab_set()
	parent.wait_window(dialog)
	return selected


def choose_masking(
	parent: tk.Misc,
	*,
	title: str,
	prompt: str,
	default_threshold: int = DEFAULT_THRESHOLD,
) -> MaskingSettings | None:
	dialog = tk.Toplevel(parent)
	dialog.title(title)
	dialog.transient(parent)
	dialog.resizable(False, False)

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
	buttons = ttk.Frame(dialog)
	buttons.pack(padx=12, pady=(6, 12))
	ttk.Button(buttons, text="Apply", command=apply_settings).pack(side="left", padx=4)
	tk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)

	dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
	dialog.bind("<Escape>", lambda _event: dialog.destroy())
	dialog.update_idletasks()
	owner = parent.winfo_toplevel()
	x = owner.winfo_rootx() + (owner.winfo_width() - dialog.winfo_width()) // 2
	y = owner.winfo_rooty() + (owner.winfo_height() - dialog.winfo_height()) // 2
	dialog.geometry(f"+{x}+{y}")
	dialog.grab_set()
	parent.wait_window(dialog)
	return selected


def choose_filter(
	parent: tk.Misc,
	*,
	title: str,
) -> FilterSettings | None:
	dialog = tk.Toplevel(parent)
	dialog.title(title)
	dialog.transient(parent)
	dialog.resizable(False, False)

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
	buttons = ttk.Frame(dialog)
	buttons.pack(padx=12, pady=(6, 12))
	ttk.Button(buttons, text="Apply", command=apply_settings).pack(side="left", padx=4)
	tk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)
	update_sigma_state()

	dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
	dialog.bind("<Escape>", lambda _event: dialog.destroy())
	dialog.update_idletasks()
	owner = parent.winfo_toplevel()
	x = owner.winfo_rootx() + (owner.winfo_width() - dialog.winfo_width()) // 2
	y = owner.winfo_rooty() + (owner.winfo_height() - dialog.winfo_height()) // 2
	dialog.geometry(f"+{x}+{y}")
	dialog.grab_set()
	parent.wait_window(dialog)
	return selected


def choose_canny(
	parent: tk.Misc,
	*,
	title: str,
) -> CannySettings | None:
	dialog = tk.Toplevel(parent)
	dialog.title(title)
	dialog.transient(parent)
	dialog.resizable(False, False)

	selected: CannySettings | None = None
	sigma_value = tk.StringVar(master=dialog, value="2.0")
	low_value = tk.StringVar(master=dialog, value="0.1")
	high_value = tk.StringVar(master=dialog, value="0.2")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			sigma = float(sigma_value.get())
			low_threshold = float(low_value.get())
			high_threshold = float(high_value.get())
		except ValueError:
			error_text.set("Enter valid numeric values.")
			return
		if sigma <= 0 or not 0 <= low_threshold < high_threshold <= 1:
			error_text.set("Use sigma > 0 and thresholds with 0 <= low < high <= 1.")
			return
		selected = CannySettings(sigma, low_threshold, high_threshold)
		dialog.destroy()

	for label, value in (
		("Sigma:", sigma_value),
		("Low threshold (0-1):", low_value),
		("High threshold (0-1):", high_value),
	):
		row = ttk.Frame(dialog)
		row.pack(fill="x", padx=16, pady=(12, 0))
		ttk.Label(row, text=label).pack(side="left")
		ttk.Spinbox(row, textvariable=value, width=8).pack(side="right")
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	buttons = ttk.Frame(dialog)
	buttons.pack(padx=12, pady=(6, 12))
	ttk.Button(buttons, text="Apply", command=apply_settings).pack(side="left", padx=4)
	tk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)

	dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
	dialog.bind("<Escape>", lambda _event: dialog.destroy())
	dialog.update_idletasks()
	owner = parent.winfo_toplevel()
	x = owner.winfo_rootx() + (owner.winfo_width() - dialog.winfo_width()) // 2
	y = owner.winfo_rooty() + (owner.winfo_height() - dialog.winfo_height()) // 2
	dialog.geometry(f"+{x}+{y}")
	dialog.grab_set()
	parent.wait_window(dialog)
	return selected


def choose_edge_visualization(
	parent: tk.Misc,
	*,
	title: str,
) -> EdgeVisualizationSettings | None:
	dialog = tk.Toplevel(parent)
	dialog.title(title)
	dialog.transient(parent)
	dialog.resizable(False, False)

	selected: EdgeVisualizationSettings | None = None
	sigma_value = tk.StringVar(master=dialog, value="2.0")
	low_value = tk.StringVar(master=dialog, value="0.1")
	high_value = tk.StringVar(master=dialog, value="0.2")
	line_width_value = tk.StringVar(master=dialog, value="0.8")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			sigma = float(sigma_value.get())
			low_threshold = float(low_value.get())
			high_threshold = float(high_value.get())
			line_width = float(line_width_value.get())
		except ValueError:
			error_text.set("Enter valid numeric values.")
			return
		if sigma <= 0 or line_width <= 0 or not 0 <= low_threshold < high_threshold <= 1:
			error_text.set("Use positive sigma and width, and 0 <= low < high <= 1.")
			return
		selected = EdgeVisualizationSettings(sigma, low_threshold, high_threshold, line_width)
		dialog.destroy()

	for label, value in (
		("Sigma:", sigma_value),
		("Low threshold (0-1):", low_value),
		("High threshold (0-1):", high_value),
		("Contour width:", line_width_value),
	):
		row = ttk.Frame(dialog)
		row.pack(fill="x", padx=16, pady=(12, 0))
		ttk.Label(row, text=label).pack(side="left")
		ttk.Spinbox(row, textvariable=value, width=8).pack(side="right")
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	buttons = ttk.Frame(dialog)
	buttons.pack(padx=12, pady=(6, 12))
	ttk.Button(buttons, text="Apply", command=apply_settings).pack(side="left", padx=4)
	tk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)

	dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
	dialog.bind("<Escape>", lambda _event: dialog.destroy())
	dialog.update_idletasks()
	owner = parent.winfo_toplevel()
	x = owner.winfo_rootx() + (owner.winfo_width() - dialog.winfo_width()) // 2
	y = owner.winfo_rooty() + (owner.winfo_height() - dialog.winfo_height()) // 2
	dialog.geometry(f"+{x}+{y}")
	dialog.grab_set()
	parent.wait_window(dialog)
	return selected


def choose_sharpening(
	parent: tk.Misc,
	*,
	title: str,
) -> SharpeningSettings | None:
	dialog = tk.Toplevel(parent)
	dialog.title(title)
	dialog.transient(parent)
	dialog.resizable(False, False)

	selected: SharpeningSettings | None = None
	radius_value = tk.StringVar(master=dialog, value="2.0")
	amount_value = tk.StringVar(master=dialog, value="1.5")
	error_text = tk.StringVar(master=dialog)

	def apply_settings() -> None:
		nonlocal selected
		try:
			radius = float(radius_value.get())
			amount = float(amount_value.get())
		except ValueError:
			error_text.set("Enter valid numeric values.")
			return
		if radius <= 0 or amount < 0:
			error_text.set("Use a positive radius and an amount of 0 or more.")
			return
		selected = SharpeningSettings(radius, amount)
		dialog.destroy()

	for label, value in (("Radius:", radius_value), ("Amount:", amount_value)):
		row = ttk.Frame(dialog)
		row.pack(fill="x", padx=16, pady=(12, 0))
		ttk.Label(row, text=label).pack(side="left")
		ttk.Spinbox(row, textvariable=value, width=8).pack(side="right")
	ttk.Label(dialog, textvariable=error_text).pack(anchor="w", padx=16, pady=(6, 0))
	buttons = ttk.Frame(dialog)
	buttons.pack(padx=12, pady=(6, 12))
	ttk.Button(buttons, text="Apply", command=apply_settings).pack(side="left", padx=4)
	tk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)

	dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
	dialog.bind("<Escape>", lambda _event: dialog.destroy())
	dialog.update_idletasks()
	owner = parent.winfo_toplevel()
	x = owner.winfo_rootx() + (owner.winfo_width() - dialog.winfo_width()) // 2
	y = owner.winfo_rooty() + (owner.winfo_height() - dialog.winfo_height()) // 2
	dialog.geometry(f"+{x}+{y}")
	dialog.grab_set()
	parent.wait_window(dialog)
	return selected