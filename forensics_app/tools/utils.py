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