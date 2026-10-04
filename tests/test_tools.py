import unittest
from unittest.mock import patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_split import ChannelSplitTool
from forensics_app.tools.channel_swap import ChannelSwapTool
from forensics_app.tools.contrast_enhancement import ContrastEnhancementTool
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.histogram import HistogramTool
from forensics_app.tools.image_info import ImageInfoTool
from forensics_app.tools.registry import ToolRegistry
from forensics_app.tools.masking import MaskingTool
from forensics_app.tools.utils import MaskingSettings


class ToolTests(unittest.TestCase):
    def test_grayscale_returns_image_without_mutating_document(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        result = GrayscaleTool().run(None, document)  # parent is unused by this tool
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(document.current.mode, "RGB")

    @patch("forensics_app.tools.channel_split.choose_channel", return_value="Green")
    def test_channel_split_extracts_selected_channel_without_mutating_document(self, _choose_channel) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1), (12, 128, 240))

        result = ChannelSplitTool().run(None, document)

        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.getpixel((0, 0)), 128)
        self.assertEqual(result.details["Channel"], "Green")
        self.assertEqual(document.current.mode, "RGB")

    @patch("forensics_app.tools.channel_split.choose_channel", return_value=None)
    def test_channel_split_returns_none_when_cancelled(self, _choose_channel) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1), "red")

        self.assertIsNone(ChannelSplitTool().run(None, document))

    @patch("forensics_app.tools.channel_split.choose_channel", return_value="Blue")
    def test_channel_split_accepts_grayscale_image(self, _choose_channel) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (1, 1), 73)

        result = ChannelSplitTool().run(None, document)

        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.getpixel((0, 0)), 73)

    @patch(
        "forensics_app.tools.channel_swap.choose_channel",
        side_effect=["Red", "Blue"],
    )
    def test_channel_swap_swaps_selected_rgb_channels(self, _askstring) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1), (10, 20, 30))

        result = ChannelSwapTool().run(None, document)

        self.assertEqual(result.image.getpixel((0, 0)), (30, 20, 10))
        self.assertEqual(result.details["Channels"], "Red and Blue")
        self.assertEqual(document.current.getpixel((0, 0)), (10, 20, 30))

    @patch(
        "forensics_app.tools.channel_swap.choose_channel",
        side_effect=["Green", "Red"],
    )
    def test_channel_swap_preserves_alpha(self, _askstring) -> None:
        document = ImageDocument()
        document.current = Image.new("RGBA", (1, 1), (10, 20, 30, 40))

        result = ChannelSwapTool().run(None, document)

        self.assertEqual(result.image.getpixel((0, 0)), (20, 10, 30, 40))

    @patch("forensics_app.tools.channel_swap.choose_channel", side_effect=["Red", "Blue"])
    def test_channel_swap_accepts_grayscale_image(self, _choose_channel) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (1, 1), 73)

        result = ChannelSwapTool().run(None, document)

        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.getpixel((0, 0)), (73, 73, 73))

    @patch(
        "forensics_app.tools.channel_swap.choose_channel",
        side_effect=["Red", "Red"],
    )
    def test_channel_swap_rejects_same_channel_twice(self, _askstring) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1), "red")

        self.assertIsNone(ChannelSwapTool().run(None, document))

    @patch(
        "forensics_app.tools.masking.choose_masking",
        return_value=MaskingSettings(True, True, False, 135),
    )
    def test_masking_creates_binary_mask(self, _choose_masking) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (4, 1))
        document.current.putdata([0, 135, 136, 255])

        result = MaskingTool().run(None, document)

        self.assertEqual(list(result.image.getdata()), [0, 0, 255, 255])
        self.assertEqual(result.details["Threshold"], 135)

    @patch(
        "forensics_app.tools.masking.choose_masking",
        return_value=MaskingSettings(False, True, False, 135),
    )
    def test_masking_keeps_only_pixels_above_threshold(self, _choose_masking) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (4, 1))
        document.current.putdata([0, 135, 136, 255])

        result = MaskingTool().run(None, document)

        self.assertEqual(list(result.image.getdata()), [0, 0, 136, 255])

    @patch(
        "forensics_app.tools.masking.choose_masking",
        return_value=MaskingSettings(False, True, False, 135),
    )
    def test_masking_accepts_grayscale_image(self, _choose_masking) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (2, 1))
        document.current.putdata([0, 200])

        result = MaskingTool().run(None, document)

        self.assertEqual(result.image.mode, "L")
        self.assertEqual(list(result.image.getdata()), [0, 200])

    @patch(
        "forensics_app.tools.masking.choose_masking",
        return_value=MaskingSettings(False, False, False, 135),
    )
    def test_masking_keeps_pixels_at_or_below_threshold(self, _choose_masking) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (4, 1))
        document.current.putdata([0, 135, 136, 255])

        result = MaskingTool().run(None, document)

        self.assertEqual(list(result.image.getdata()), [0, 135, 0, 0])

    @patch(
        "forensics_app.tools.masking.choose_masking",
        return_value=MaskingSettings(False, True, False, 200),
    )
    def test_masking_uses_selected_threshold(self, _choose_masking) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (4, 1))
        document.current.putdata([199, 200, 201, 255])

        result = MaskingTool().run(None, document)

        self.assertEqual(list(result.image.getdata()), [0, 0, 201, 255])
        self.assertEqual(result.details["Threshold"], 200)

    def test_registry_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            ToolRegistry([GrayscaleTool(), GrayscaleTool()])

    def test_histogram_renders_rgb_curves_without_mutating_document(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (2, 2))
        document.current.putdata([(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)])

        result = HistogramTool().run(None, document)

        self.assertEqual(result.image.size, (800, 500))
        self.assertEqual(result.details["Peak frequency"], 3)
        self.assertEqual(document.current.getpixel((1, 0)), (255, 0, 0))

    def test_histogram_accepts_grayscale_image(self) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (2, 2), 128)

        result = HistogramTool().run(None, document)

        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.details["Peak frequency"], 4)

    def test_image_info_accepts_grayscale_image(self) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (2, 3), 128)

        result = ImageInfoTool().run(None, document)

        self.assertEqual(result.details["Mode"], "L")
        self.assertEqual(result.details["Width"], "2 px")
        self.assertEqual(result.details["Height"], "3 px")

    def test_contrast_enhancement_methods_return_same_size_and_preserve_source(self) -> None:
        document = ImageDocument()
        document.current = Image.frombytes("L", (16, 16), bytes(range(256)))

        for method, label in (
            ("low_contrast", "Create low-contrast image"),
            ("stretch", "Contrast stretching"),
            ("equalize", "Histogram equalization"),
            ("clahe", "Adaptive histogram equalization (CLAHE)"),
        ):
            with self.subTest(method=method):
                with patch(
                    "forensics_app.tools.contrast_enhancement.choose_option",
                    return_value=method,
                ):
                    result = ContrastEnhancementTool().run(None, document)

                self.assertEqual(result.image.mode, "L")
                self.assertEqual(result.image.size, document.current.size)
                self.assertEqual(result.details["Operation"], label)
                self.assertEqual(document.current.getpixel((15, 15)), 255)

    def test_contrast_enhancement_preserves_color_channels(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (16, 16))
        document.current.putdata(
            [(value, 255 - value, 64 + value // 2) for value in range(256)]
        )

        for method in ("low_contrast", "stretch", "equalize", "clahe"):
            with self.subTest(method=method):
                with patch(
                    "forensics_app.tools.contrast_enhancement.choose_option",
                    return_value=method,
                ):
                    result = ContrastEnhancementTool().run(None, document)

                pixel = result.image.getpixel((8, 8))
                self.assertEqual(result.image.mode, "RGB")
                self.assertGreater(len(set(pixel)), 1)
                self.assertEqual(document.current.getpixel((8, 8)), (136, 119, 132))


if __name__ == "__main__":
    unittest.main()
