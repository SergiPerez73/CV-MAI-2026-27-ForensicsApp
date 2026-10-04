"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .contrast_enhancement import ContrastEnhancementTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .image_info import ImageInfoTool
from .masking import MaskingTool
from .registry import ToolRegistry


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
            MaskingTool(),
            ContrastEnhancementTool(),
            HistogramTool(),
        ]
    )


__all__ = [
    "ToolRegistry",
    "build_tool_registry",
    "ChannelSplitTool",
    "ChannelSwapTool",
    "ContrastEnhancementTool",
    "HistogramTool",
    "MaskingTool",
]
