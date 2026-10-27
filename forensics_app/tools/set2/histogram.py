"""Visualize the pixel-frequency histogram for each RGB channel."""

from __future__ import annotations

import tkinter as tk

from PIL import Image, ImageDraw

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "View channel histogram"
    category = "Set 2"
    description = "Plot red, green, and blue pixel frequencies across intensities 0–255."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window
        source = document.current
        grayscale = len(source.getbands()) == 1
        if grayscale:
            histograms = [source.convert("L").histogram()]
            channel_colors = (("Grayscale", "#ffffff"),)
            background_color = "#20252b"
            text_color = "#ffffff"
            axis_color = "#adb5bd"
            grid_color = "#4d565f"
        else:
            source = source.convert("RGB")
            histograms = [source.histogram()[index * 256 : (index + 1) * 256] for index in range(3)]
            channel_colors = (("Red", "#d83b3b"), ("Green", "#16864b"), ("Blue", "#3478c8"))
            background_color = "#ffffff"
            text_color = "#20252b"
            axis_color = "#69737d"
            grid_color = "#e2e6ea"

        chart = Image.new("RGB", (800, 500), background_color)
        draw = ImageDraw.Draw(chart)
        left, top, right, bottom = 72, 76, 780, 424
        plot_width = right - left
        plot_height = bottom - top
        peak = max(max(channel) for channel in histograms)

        title = "Grayscale histogram" if grayscale else "RGB channel histogram"
        draw.text((left, 18), title, fill=text_color)
        for index, (name, color) in enumerate(channel_colors):
            legend_x = left + index * 125
            draw.line((legend_x, 51, legend_x + 20, 51), fill=color, width=3)
            draw.text((legend_x + 27, 44), name, fill=text_color)

        for division in range(5):
            y = bottom - round(plot_height * division / 4)
            frequency = round(peak * division / 4)
            draw.line((left, y, right, y), fill=grid_color, width=1)
            draw.text((8, y - 6), str(frequency), fill=axis_color)

        draw.rectangle((left, top, right, bottom), outline=axis_color, width=1)
        for intensity in (0, 64, 128, 192, 255):
            x = left + round(plot_width * intensity / 255)
            draw.line((x, bottom, x, bottom + 4), fill=axis_color, width=1)
            draw.text((x - 8, bottom + 10), str(intensity), fill=axis_color)

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
            message=f"Displayed the {'grayscale' if grayscale else 'RGB channel'} histogram.",
            details={
                "Operation": "Grayscale histogram" if grayscale else "Channel histogram",
                "Image size": f"{source.width} x {source.height}",
                "Pixels": source.width * source.height,
                "Peak frequency": peak,
            },
        )