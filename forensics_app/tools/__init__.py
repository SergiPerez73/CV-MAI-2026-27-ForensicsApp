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
from .hog import HOGTool
from .image_sharpening import ImageSharpeningTool
from .image_info import ImageInfoTool
from .masking import MaskingTool
from .normalized_cross_correlation import NormalizedCrossCorrelationTool
from .person_detection import PersonDetectionTool
from .registry import ToolRegistry
from .template_matching import TemplateMatchingTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            GrayscaleTool(),
            ImageInfoTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
            MaskingTool(),
            HistogramTool(),
            ContrastEnhancementTool(),
            CannyEdgeDetectionTool(),
            ConvolutionTool(),
            FiltersTool(),
            EdgeVisualizationTool(),
            ImageSharpeningTool(),
            NormalizedCrossCorrelationTool(),
            TemplateMatchingTool(),
            HOGTool(),
            PersonDetectionTool(),
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
    "HOGTool",
    "ImageSharpeningTool",
    "MaskingTool",
    "NormalizedCrossCorrelationTool",
    "PersonDetectionTool",
    "TemplateMatchingTool",
]
