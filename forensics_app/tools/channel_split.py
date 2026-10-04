"""Extract one color channel from an image."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .constants import CHANNELS
from .utils import choose_channel


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Split color channel"
    category = "Image processing"
    description = "Extract the red, green, or blue channel as a grayscale image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        channel = choose_channel(
            parent,
            title="Split color channel",
            prompt="Choose a channel to extract:",
        )
        if channel is None:
            return None

        image_numpy = np.array(document.current.convert("RGB"))
        output_image = Image.fromarray(image_numpy[:, :, CHANNELS[channel]])
        
        return ToolResult(
            image=output_image,
            message=f"Extracted the {channel.lower()} channel.",
            details={
                "Operation": "Channel split",
                "Channel": channel,
                "Output mode": output_image.mode,
            },
        )
