"""Visualize the pixel-frequency histogram for each RGB channel."""

from __future__ import annotations

import tkinter as tk

from PIL import Image, ImageDraw

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "View channel histogram"
    category = "Image analysis"
    description = "Plot red, green, and blue pixel frequencies across intensities 0–255."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window
        source = document.current.convert("RGB")
        histograms = [source.histogram()[index * 256 : (index + 1) * 256] for index in range(3)]
        channel_colors = (("Red", "#d83b3b"), ("Green", "#16864b"), ("Blue", "#3478c8"))

        chart = Image.new("RGB", (800, 500), "#ffffff")
        draw = ImageDraw.Draw(chart)
        left, top, right, bottom = 72, 76, 780, 424
        plot_width = right - left
        plot_height = bottom - top
        peak = max(max(channel) for channel in histograms)

        draw.text((left, 18), "RGB channel histogram", fill="#20252b")
        for index, (name, color) in enumerate(channel_colors):
            legend_x = left + index * 125
            draw.line((legend_x, 51, legend_x + 20, 51), fill=color, width=3)
            draw.text((legend_x + 27, 44), name, fill="#343a40")

        for division in range(5):
            y = bottom - round(plot_height * division / 4)
            frequency = round(peak * division / 4)
            draw.line((left, y, right, y), fill="#e2e6ea", width=1)
            draw.text((8, y - 6), str(frequency), fill="#59636e")

        draw.rectangle((left, top, right, bottom), outline="#69737d", width=1)
        for intensity in (0, 64, 128, 192, 255):
            x = left + round(plot_width * intensity / 255)
            draw.line((x, bottom, x, bottom + 4), fill="#69737d", width=1)
            draw.text((x - 8, bottom + 10), str(intensity), fill="#59636e")

        for channel, (_, color) in zip(histograms, channel_colors):
            points = [
                (
                    left + round(plot_width * intensity / 255),
                    bottom - round(plot_height * count / peak),
                )
                for intensity, count in enumerate(channel)
            ]
            draw.line(points, fill=color, width=2)

        return ToolResult(
            image=chart,
            message="Displayed the RGB channel histogram.",
            details={
                "Operation": "Channel histogram",
                "Image size": f"{source.width} x {source.height}",
                "Pixels": source.width * source.height,
                "Peak frequency": peak,
            },
        )