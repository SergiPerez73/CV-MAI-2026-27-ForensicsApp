"""Canvas that displays a PIL image scaled to the available space."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


class ImageView(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=8)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            self,
            background="#20242b",
            borderwidth=0,
            highlightthickness=0,
        )
        self.vertical_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.horizontal_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(
            xscrollcommand=self.horizontal_scrollbar.set,
            yscrollcommand=self.vertical_scrollbar.set,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        self.horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        self.vertical_scrollbar.grid_remove()
        self.horizontal_scrollbar.grid_remove()
        self.canvas.bind("<Configure>", self._on_resize)
        self._source: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._resize_job: str | None = None
        self._zoom: float | None = None
        self.show(None)

    def show(self, image: Image.Image | None) -> None:
        self._source = image
        self._render()

    def set_zoom(self, zoom: float | None) -> None:
        self._zoom = zoom
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self._render()

    def _on_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(50, self._render)

    def _render(self) -> None:
        self._resize_job = None
        self.canvas.delete("all")
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        if self._source is None:
            self.canvas.create_text(
                canvas_width // 2,
                canvas_height // 2,
                text="Open an image to begin",
                fill="#c7ccd4",
                font=("TkDefaultFont", 16),
            )
            self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
            self.vertical_scrollbar.grid_remove()
            self.horizontal_scrollbar.grid_remove()
            self._photo = None
            return

        preview = self._source.copy()
        if self._zoom is None:
            preview.thumbnail((max(canvas_width - 24, 1), max(canvas_height - 24, 1)), Image.Resampling.LANCZOS)
        else:
            preview = preview.resize(
                (round(preview.width * self._zoom), round(preview.height * self._zoom)),
                Image.Resampling.LANCZOS,
            )
        content_width = max(canvas_width, preview.width)
        content_height = max(canvas_height, preview.height)
        self.canvas.configure(scrollregion=(0, 0, content_width, content_height))
        if preview.width > canvas_width:
            self.horizontal_scrollbar.grid()
        else:
            self.horizontal_scrollbar.grid_remove()
        if preview.height > canvas_height:
            self.vertical_scrollbar.grid()
        else:
            self.vertical_scrollbar.grid_remove()
        self._photo = ImageTk.PhotoImage(preview)
        self.canvas.create_image(
            (content_width - preview.width) // 2,
            (content_height - preview.height) // 2,
            image=self._photo,
            anchor="nw",
        )
