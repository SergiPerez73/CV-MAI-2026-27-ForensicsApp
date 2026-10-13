import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_split import ChannelSplitTool
from forensics_app.tools.channel_swap import ChannelSwapTool
from forensics_app.tools.canny_edge_detection import CannyEdgeDetectionTool, CannySettings
from forensics_app.tools.contrast_enhancement import ContrastEnhancementTool
from forensics_app.tools.convolution import ConvolutionTool, ConvolutionSettings
from forensics_app.tools.edge_visualization import EdgeVisualizationTool, EdgeVisualizationSettings
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.filters import FiltersTool, FilterSettings
from forensics_app.tools.histogram import HistogramTool
from forensics_app.tools.image_sharpening import ImageSharpeningTool, SharpeningSettings
from forensics_app.tools.image_info import ImageInfoTool
from forensics_app.tools.registry import ToolRegistry
from forensics_app.tools.masking import MaskingTool, MaskingSettings
from forensics_app.tools.utils import (
    add_dialog_buttons,
    add_numeric_field,
    choose_option,
    create_dialog,
    wait_for_dialog,
)


class DialogHelperTests(unittest.TestCase):
    @patch("forensics_app.tools.utils.tk.Toplevel")
    def test_create_dialog_configures_owner_and_title(self, toplevel) -> None:
        parent = MagicMock()

        dialog = create_dialog(parent, title="Settings")

        self.assertIs(dialog, toplevel.return_value)
        toplevel.assert_called_once_with(parent)
        dialog.title.assert_called_once_with("Settings")
        dialog.transient.assert_called_once_with(parent)
        dialog.resizable.assert_called_once_with(False, False)

    @patch("forensics_app.tools.utils.tk.Button")
    @patch("forensics_app.tools.utils.ttk.Button")
    @patch("forensics_app.tools.utils.ttk.Frame")
    def test_dialog_buttons_use_apply_and_destroy_callbacks(self, frame, apply_button, cancel_button) -> None:
        dialog = MagicMock()
        apply_settings = MagicMock()

        add_dialog_buttons(dialog, apply_settings)

        apply_button.call_args.kwargs["command"]()
        cancel_button.call_args.kwargs["command"]()
        apply_settings.assert_called_once_with()
        dialog.destroy.assert_called_once_with()
        frame.assert_called_once_with(dialog)

    @patch("forensics_app.tools.utils.ttk.Spinbox")
    @patch("forensics_app.tools.utils.ttk.Label")
    @patch("forensics_app.tools.utils.ttk.Frame")
    def test_numeric_field_uses_supplied_variable(self, frame, label, spinbox) -> None:
        dialog = MagicMock()
        value = MagicMock()

        control = add_numeric_field(dialog, "Sigma:", value)

        self.assertIs(control, spinbox.return_value)
        label.assert_called_once_with(frame.return_value, text="Sigma:")
        spinbox.assert_called_once_with(frame.return_value, textvariable=value, width=8)

    def test_wait_for_dialog_centers_and_handles_cancellation(self) -> None:
        parent = MagicMock()
        owner = parent.winfo_toplevel.return_value
        owner.winfo_rootx.return_value = 100
        owner.winfo_rooty.return_value = 50
        owner.winfo_width.return_value = 800
        owner.winfo_height.return_value = 600
        dialog = MagicMock()
        dialog.winfo_width.return_value = 200
        dialog.winfo_height.return_value = 100

        wait_for_dialog(parent, dialog)

        dialog.geometry.assert_called_once_with("+400+300")
        dialog.protocol.assert_called_once_with("WM_DELETE_WINDOW", dialog.destroy)
        dialog.grab_set.assert_called_once_with()
        parent.wait_window.assert_called_once_with(dialog)
        dialog.bind.call_args.args[1](None)
        dialog.destroy.assert_called_once_with()

    @patch("forensics_app.tools.utils.wait_for_dialog")
    @patch("forensics_app.tools.utils.create_dialog")
    @patch("forensics_app.tools.utils.ttk.Frame")
    @patch("forensics_app.tools.utils.ttk.Label")
    @patch("forensics_app.tools.utils.ttk.Button")
    def test_choose_option_returns_selection_and_disables_excluded_options(
        self, button, _label, _frame, create, wait
    ) -> None:
        wait.side_effect = lambda *_args: button.call_args_list[1].kwargs["command"]()
        parent = MagicMock()

        result = choose_option(
            parent, title="Channel", prompt="Choose:",
            options={"red": "Red", "blue": "Blue"}, disabled_options={"red"},
        )

        self.assertEqual(result, "blue")
        self.assertEqual(button.call_args_list[0].kwargs["state"], "disabled")
        self.assertEqual(button.call_args_list[1].kwargs["state"], "normal")
        create.return_value.destroy.assert_called_once_with()
        wait.side_effect = None
        self.assertIsNone(choose_option(parent, title="Channel", prompt="Choose:", options={"red": "Red"}))


class ToolTests(unittest.TestCase):
    @patch(
        "forensics_app.tools.image_sharpening.choose_sharpening",
        return_value=SharpeningSettings(2.0, 1.5),
    )
    @patch("forensics_app.tools.image_sharpening.filters.unsharp_mask")
    def test_image_sharpening_uses_selected_radius_and_amount(
        self, unsharp_mask, _choose_sharpening
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (2, 1))
        document.current.putdata([0, 255])
        unsharp_mask.return_value = np.array([[0.0, 1.0]])

        result = ImageSharpeningTool().run(None, document)

        self.assertEqual(unsharp_mask.call_args.kwargs, {
            "radius": 2.0,
            "amount": 1.5,
            "channel_axis": None,
        })
        self.assertEqual(list(result.image.getdata()), [0, 255])
        self.assertEqual(result.details["Radius"], 2.0)
        self.assertEqual(result.details["Amount"], 1.5)

    @patch(
        "forensics_app.tools.edge_visualization.choose_edge_visualization",
        return_value=EdgeVisualizationSettings(2.0, 0.1, 0.2, 2.0),
    )
    @patch(
        "forensics_app.tools.edge_visualization.feature.canny",
        return_value=np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        ),
    )
    def test_edge_visualization_draws_red_contours_over_original(
        self, _canny, _choose_canny
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (3, 3), (40, 80, 120))

        result = EdgeVisualizationTool().run(None, document)

        self.assertEqual(result.image.size, document.current.size)
        self.assertEqual(result.details["Operation"], "Canny contour overlay")
        self.assertEqual(result.details["Contour width"], 2.0)

    @patch(
        "forensics_app.tools.canny_edge_detection.choose_canny",
        return_value=CannySettings(2.0, 0.1, 0.2),
    )
    @patch(
        "forensics_app.tools.canny_edge_detection.feature.canny",
        return_value=np.array([[False, True]]),
    )
    def test_canny_uses_selected_parameters_and_returns_binary_edges(
        self, canny, _choose_canny
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (2, 1))
        document.current.putdata([0, 255])

        result = CannyEdgeDetectionTool().run(None, document)

        image = canny.call_args.args[0]
        self.assertEqual(image.tolist(), [[0.0, 1.0]])
        self.assertEqual(canny.call_args.kwargs, {
            "sigma": 2.0,
            "low_threshold": 0.1,
            "high_threshold": 0.2,
        })
        self.assertEqual(list(result.image.getdata()), [0, 255])
        self.assertEqual(result.image.mode, "L")

    def test_filters_apply_selected_operations_and_gaussian_sigma(self) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (9, 9))
        document.current.putpixel((4, 4), 255)

        for name in ("gaussian", "median", "sobel"):
            with self.subTest(filter=name):
                settings = FilterSettings(name, 2.0)
                with patch(
                    "forensics_app.tools.filters.choose_filter",
                    return_value=settings,
                ):
                    result = FiltersTool().run(None, document)

                self.assertEqual(result.image.mode, "L")
                self.assertEqual(result.image.size, document.current.size)
                self.assertEqual(result.details["Operation"], {
                    "gaussian": "Gaussian smoothing",
                    "median": "Median denoising",
                    "sobel": "Sobel edge detection",
                }[name])

        with patch(
            "forensics_app.tools.filters.choose_filter",
            return_value=FilterSettings("gaussian", 0.5),
        ):
            low_sigma = FiltersTool().run(None, document)
        with patch(
            "forensics_app.tools.filters.choose_filter",
            return_value=FilterSettings("gaussian", 2.0),
        ):
            high_sigma = FiltersTool().run(None, document)

        self.assertNotEqual(list(low_sigma.image.getdata()), list(high_sigma.image.getdata()))
        self.assertEqual(high_sigma.details["Sigma"], 2.0)

    def test_convolution_uses_selected_kernel_shape(self) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (5, 5))
        document.current.putdata([0] * 12 + [255] + [0] * 12)

        for direction, shape in (
            ("horizontal", "1 x 3"),
            ("vertical", "3 x 1"),
            ("both", "3 x 3"),
        ):
            with self.subTest(direction=direction):
                with patch(
                    "forensics_app.tools.convolution.choose_convolution",
                    return_value=ConvolutionSettings(3, direction),
                ):
                    result = ConvolutionTool().run(None, document)

                self.assertEqual(result.image.mode, "L")
                self.assertEqual(result.image.size, document.current.size)
                self.assertEqual(result.details["Kernel shape"], shape)
                self.assertGreater(max(result.image.getdata()), 0)

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
    @patch("forensics_app.tools.masking.filedialog.askopenfilename", return_value="foreground.png")
    @patch("forensics_app.tools.masking.Image.open")
    def test_masking_composites_binary_foreground_over_current_image(
        self, open_image, _choose_file, _choose_masking
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 1), "red")
        foreground = Image.new("L", (4, 1))
        foreground.putdata([0, 135, 136, 255])
        opened_image = MagicMock()
        opened_image.__enter__.return_value = foreground
        open_image.return_value = opened_image

        result = MaskingTool().run(None, document)

        self.assertEqual(list(result.image.getdata()), [(255, 0, 0), (255, 0, 0), (255, 255, 255), (255, 255, 255)])
        self.assertEqual(result.details["Threshold"], 135)
        self.assertEqual(result.details["Foreground"], "foreground.png")

    @patch(
        "forensics_app.tools.masking.choose_masking",
        return_value=MaskingSettings(False, True, False, 135),
    )
    @patch("forensics_app.tools.masking.filedialog.askopenfilename", return_value="foreground.png")
    @patch("forensics_app.tools.masking.Image.open")
    def test_masking_preserves_selected_foreground_pixels(
        self, open_image, _choose_file, _choose_masking
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 1), "blue")
        foreground = Image.new("L", (4, 1))
        foreground.putdata([0, 135, 136, 255])
        opened_image = MagicMock()
        opened_image.__enter__.return_value = foreground
        open_image.return_value = opened_image

        result = MaskingTool().run(None, document)

        self.assertEqual(
            list(result.image.getdata()),
            [(0, 0, 255), (0, 0, 255), (136, 136, 136), (255, 255, 255)],
        )

    @patch(
        "forensics_app.tools.masking.choose_masking",
        return_value=MaskingSettings(False, True, False, 135),
    )
    @patch("forensics_app.tools.masking.filedialog.askopenfilename", return_value="foreground.png")
    @patch("forensics_app.tools.masking.Image.open")
    def test_masking_shows_grayscale_background_through_transparent_pixels(
        self, open_image, _choose_file, _choose_masking
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (2, 1))
        document.current.putdata([10, 20])
        foreground = Image.new("L", (2, 1))
        foreground.putdata([0, 200])
        opened_image = MagicMock()
        opened_image.__enter__.return_value = foreground
        open_image.return_value = opened_image

        result = MaskingTool().run(None, document)

        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(list(result.image.getdata()), [(10, 10, 10), (200, 200, 200)])

    @patch(
        "forensics_app.tools.masking.choose_masking",
        return_value=MaskingSettings(False, False, False, 135),
    )
    @patch("forensics_app.tools.masking.filedialog.askopenfilename", return_value="foreground.png")
    @patch("forensics_app.tools.masking.Image.open")
    def test_masking_can_keep_dark_foreground_pixels(
        self, open_image, _choose_file, _choose_masking
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 1), "blue")
        foreground = Image.new("L", (4, 1))
        foreground.putdata([0, 135, 136, 255])
        opened_image = MagicMock()
        opened_image.__enter__.return_value = foreground
        open_image.return_value = opened_image

        result = MaskingTool().run(None, document)

        self.assertEqual(
            list(result.image.getdata()),
            [(0, 0, 0), (135, 135, 135), (0, 0, 255), (0, 0, 255)],
        )

    @patch(
        "forensics_app.tools.masking.choose_masking",
        return_value=MaskingSettings(False, True, False, 200),
    )
    @patch("forensics_app.tools.masking.filedialog.askopenfilename", return_value="foreground.png")
    @patch("forensics_app.tools.masking.Image.open")
    def test_masking_uses_selected_threshold(self, open_image, _choose_file, _choose_masking) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 1), "blue")
        foreground = Image.new("L", (4, 1))
        foreground.putdata([199, 200, 201, 255])
        opened_image = MagicMock()
        opened_image.__enter__.return_value = foreground
        open_image.return_value = opened_image

        result = MaskingTool().run(None, document)

        self.assertEqual(
            list(result.image.getdata()),
            [(0, 0, 255), (0, 0, 255), (201, 201, 201), (255, 255, 255)],
        )
        self.assertEqual(result.details["Threshold"], 200)

    @patch(
        "forensics_app.tools.masking.choose_masking",
        return_value=MaskingSettings(False, True, False, 135, True),
    )
    @patch("forensics_app.tools.masking.filedialog.askopenfilename", return_value="foreground.png")
    @patch("forensics_app.tools.masking.Image.open")
    def test_masking_can_select_foreground_pixels_from_background_brightness(
        self, open_image, _choose_file, _choose_masking
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (2, 1))
        document.current.putdata([10, 200])
        foreground = Image.new("L", (2, 1))
        foreground.putdata([250, 50])
        opened_image = MagicMock()
        opened_image.__enter__.return_value = foreground
        open_image.return_value = opened_image

        result = MaskingTool().run(None, document)

        self.assertEqual(list(result.image.getdata()), [(10, 10, 10), (50, 50, 50)])
        self.assertEqual(result.message, "Applied background-brightness mask to foreground.png.")
        self.assertEqual(result.details["Operation"], "Masking composite")
        self.assertEqual(result.details["Mask source"], "Background")

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
        self.assertEqual(result.details["Operation"], "Grayscale histogram")
        self.assertEqual(result.image.getpixel((427, 76)), (255, 255, 255))

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
