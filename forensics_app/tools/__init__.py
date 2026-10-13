"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .canny_edge_detection import CannyEdgeDetectionTool
from .contrast_enhancement import ContrastEnhancementTool
from .convolution import ConvolutionTool
from .edge_visualization import EdgeVisualizationTool
from .filters import FiltersTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .image_sharpening import ImageSharpeningTool
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
            CannyEdgeDetectionTool(),
            ConvolutionTool(),
            FiltersTool(),
            EdgeVisualizationTool(),
            ImageSharpeningTool(),
            ContrastEnhancementTool(),
            HistogramTool(),
        ]
    )


__all__ = [
    "ToolRegistry",
    "build_tool_registry",
    "ChannelSplitTool",
    "ChannelSwapTool",
    "CannyEdgeDetectionTool",
    "ContrastEnhancementTool",
    "ConvolutionTool",
    "EdgeVisualizationTool",
    "FiltersTool",
    "HistogramTool",
    "ImageSharpeningTool",
    "MaskingTool",
]
