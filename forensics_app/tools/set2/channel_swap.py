"""Swap two color channels in an image."""

from __future__ import annotations

import numpy as np
from PIL import Image

import tkinter as tk

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..constants import CHANNELS
from ..utils import choose_channel

class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Swap color channels"
    category = "Set 2"
    description = "Swap the values of any two color channels in the working image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        first_name = choose_channel(
            parent,
            title="Swap color channels",
            prompt="Choose the first channel:",
        )
        if first_name is None:
            return None
        first_index = CHANNELS.get(first_name)

        second_name = choose_channel(
            parent,
            title="Swap color channels",
            prompt=f"Choose a different channel (not {first_name}):",
            excluded_channels={first_name},
        )
        if second_name is None:
            return None
        second_index = CHANNELS.get(second_name)

        image_numpy = np.array(document.current)
        if image_numpy.ndim == 2:
            image_numpy = np.array(document.current.convert("RGB"))
        image_numpy_copy = image_numpy.copy()
        image_numpy[:,:,first_index] = image_numpy_copy[:,:,second_index]
        image_numpy[:,:,second_index] = image_numpy_copy[:,:,first_index]
        output_image = Image.fromarray(image_numpy)

        return ToolResult(
            image=output_image,
            message=f"Swapped the {first_name.lower()} and {second_name.lower()} channels.",
            details={
                "Operation": "Channel swap",
                "Channels": f"{first_name} and {second_name}",
                "Output mode": output_image.mode,
            },
        )