from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from .constants import CHANNELS


def create_dialog(parent: tk.Misc, *, title: str) -> tk.Toplevel:
	dialog = tk.Toplevel(parent)
	dialog.title(title)
	dialog.transient(parent)
	dialog.resizable(False, False)
	return dialog


def add_dialog_buttons(dialog: tk.Toplevel, apply_settings: Callable[[], None]) -> None:
	buttons = ttk.Frame(dialog)
	buttons.pack(padx=12, pady=(6, 12))
	ttk.Button(buttons, text="Apply", command=apply_settings).pack(side="left", padx=4)
	tk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)


def add_numeric_field(dialog: tk.Toplevel, label: str, value: tk.StringVar) -> ttk.Spinbox:
	row = ttk.Frame(dialog)
	row.pack(fill="x", padx=16, pady=(12, 0))
	ttk.Label(row, text=label).pack(side="left")
	control = ttk.Spinbox(row, textvariable=value, width=8)
	control.pack(side="right")
	return control


def wait_for_dialog(parent: tk.Misc, dialog: tk.Toplevel) -> None:
	dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
	dialog.bind("<Escape>", lambda _event: dialog.destroy())
	dialog.update_idletasks()
	owner = parent.winfo_toplevel()
	x = owner.winfo_rootx() + (owner.winfo_width() - dialog.winfo_width()) // 2
	y = owner.winfo_rooty() + (owner.winfo_height() - dialog.winfo_height()) // 2
	dialog.geometry(f"+{x}+{y}")
	dialog.grab_set()
	parent.wait_window(dialog)


def choose_option(
	parent: tk.Misc,
	*,
	title: str,
	prompt: str,
	options: dict[str, str],
	disabled_options: set[str] | None = None,
) -> str | None:
	dialog = create_dialog(parent, title=title)

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

	wait_for_dialog(parent, dialog)
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


def figure_to_image(figure: Figure) -> Image.Image:
	"""Render a Matplotlib figure as an RGB PIL image."""
	canvas = FigureCanvasAgg(figure)
	canvas.draw()
	return Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())