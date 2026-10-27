"""Register course functionality here so it appears in the sidebar."""

from .set1.grayscale import GrayscaleTool
from .set1.image_info import ImageInfoTool
from .set2.channel_split import ChannelSplitTool
from .set2.channel_swap import ChannelSwapTool
from .set2.contrast_enhancement import ContrastEnhancementTool
from .set2.histogram import HistogramTool
from .set2.masking import MaskingTool
from .set3.canny_edge_detection import CannyEdgeDetectionTool
from .set3.convolution import ConvolutionTool
from .set3.edge_visualization import EdgeVisualizationTool
from .set3.filters import FiltersTool
from .set3.image_sharpening import ImageSharpeningTool
from .set4.hog import HOGTool
from .set4.normalized_cross_correlation import NormalizedCrossCorrelationTool
from .set4.person_detection import PersonDetectionTool
from .set4.template_matching import TemplateMatchingTool
from .set5.closest_image_lookup import ClosestImageLookupTool
from .set5.corner_detection import CornerDetectionTool
from .set5.image_matching import ImageMatchingTool
from .set5.keypoint_descriptors import KeypointDescriptorsTool
from .registry import ToolRegistry


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            CornerDetectionTool(),
            KeypointDescriptorsTool(),
            ImageMatchingTool(),
            ClosestImageLookupTool(),
            NormalizedCrossCorrelationTool(),
            TemplateMatchingTool(),
            HOGTool(),
            PersonDetectionTool(),
            CannyEdgeDetectionTool(),
            ConvolutionTool(),
            FiltersTool(),
            EdgeVisualizationTool(),
            ImageSharpeningTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
            MaskingTool(),
            HistogramTool(),
            ContrastEnhancementTool(),
            GrayscaleTool(),
            ImageInfoTool(),
        ]
    )


__all__ = [
    "ToolRegistry",
    "build_tool_registry",
    "ChannelSplitTool",
    "ChannelSwapTool",
    "CannyEdgeDetectionTool",
    "ClosestImageLookupTool",
    "CornerDetectionTool",
    "ContrastEnhancementTool",
    "ConvolutionTool",
    "EdgeVisualizationTool",
    "FiltersTool",
    "HistogramTool",
    "HOGTool",
    "ImageSharpeningTool",
    "ImageMatchingTool",
    "KeypointDescriptorsTool",
    "MaskingTool",
    "NormalizedCrossCorrelationTool",
    "PersonDetectionTool",
    "TemplateMatchingTool",
]
