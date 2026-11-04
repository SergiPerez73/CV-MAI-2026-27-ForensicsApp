import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import cv2
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.set2.channel_split import ChannelSplitTool
from forensics_app.tools.set2.channel_swap import ChannelSwapTool
from forensics_app.tools.set3.canny_edge_detection import CannyEdgeDetectionTool, CannySettings
from forensics_app.tools.set5.closest_image_lookup import ClosestImageLookupTool
from forensics_app.tools.set5.corner_detection import CornerDetectionTool, CornerSettings
from forensics_app.tools.set2.contrast_enhancement import ContrastEnhancementTool
from forensics_app.tools.set3.convolution import ConvolutionTool, ConvolutionSettings
from forensics_app.tools.set3.edge_visualization import EdgeVisualizationTool, EdgeVisualizationSettings
from forensics_app.tools.set5.utils import DescriptorSettings, MatchingSettings, extract_features, match_features
from forensics_app.tools.set1.grayscale import GrayscaleTool
from forensics_app.tools.set3.filters import FiltersTool, FilterSettings
from forensics_app.tools.set2.histogram import HistogramTool
from forensics_app.tools.set4.hog import HOGSettings, HOGTool
from forensics_app.tools.set3.image_sharpening import ImageSharpeningTool, SharpeningSettings
from forensics_app.tools.set1.image_info import ImageInfoTool
from forensics_app.tools.set5.image_matching import ImageMatchingTool
from forensics_app.tools.set5.keypoint_descriptors import KeypointDescriptorsTool
from forensics_app.tools.registry import ToolRegistry
from forensics_app.tools.set2.masking import MaskingTool, MaskingSettings
from forensics_app.tools.set4.normalized_cross_correlation import NormalizedCrossCorrelationTool
from forensics_app.tools.set4.person_detection import PersonDetectionSettings, PersonDetectionTool
from forensics_app.tools.set4.template_matching import TemplateMatchingSettings, TemplateMatchingTool
from forensics_app.tools.set6.face_detection import FaceDetectionSettings, FaceDetectionTool, detect_faces
from forensics_app.tools.set7.codebook import (
    CLASSES, CodebookConstructionTool, CodebookSettings, build_codebook, compute_histograms,
    CodebookModel, extract_phow, extract_pyramid_sift, extract_sift, load_codebook, render_codebook, split_pyramid,
)
from forensics_app.tools.set7.histogram import VisualWordHistogramTool, render_histogram
from forensics_app.tools.set7.classification import (
    VisualWordClassifier, VisualWordPredictionTool, load_classifier, predict_image, save_classifier, train_classifier,
)
from forensics_app.tools.utils import (
    add_dialog_buttons,
    add_numeric_field,
    choose_option,
    create_dialog,
    wait_for_dialog,
)


class CodebookTests(unittest.TestCase):
    def test_registered_without_requiring_an_image(self) -> None:
        tools = dict(build_tool_registry().categories())["Set 7"]
        self.assertTrue(any(isinstance(tool, CodebookConstructionTool) for tool in tools))
        self.assertFalse(CodebookConstructionTool.requires_image)

    def test_sift_uses_configured_dense_grid_octaves_and_scales(self) -> None:
        pixels = np.random.default_rng(42).integers(0, 256, (64, 64), dtype=np.uint8)
        settings = CodebookSettings(step=8, sift_size=4, n_octaves=2, n_scales=2)
        keypoints, descriptors = extract_sift(pixels, settings)
        self.assertEqual(descriptors.shape, (len(keypoints), 128))
        self.assertEqual(descriptors.dtype, np.float32)
        self.assertTrue(all(point.angle == 0 for point in keypoints))
        self.assertEqual({point.octave for point in keypoints}, {0, 1, 1 << 8, 1 | (1 << 8)})
        first_scale = [point for point in keypoints if point.octave == 0]
        self.assertEqual({point.pt[0] for point in first_scale}, set(range(6, 58, 8)))
        np.testing.assert_allclose([point.size for point in first_scale], 8 / 3)

    def test_pyramid_splits_into_21_regions_without_losing_border_pixels(self) -> None:
        image = np.arange(65 * 67).reshape(65, 67)
        regions = split_pyramid(image, (1, 4, 16))
        self.assertEqual(len(regions), 21)
        np.testing.assert_array_equal(regions[0][0], image)
        offset = 0
        for count in (1, 4, 16):
            reconstructed = np.zeros_like(image)
            for region, (left, top) in regions[offset:offset + count]:
                height, width = region.shape
                reconstructed[top:top + height, left:left + width] = region
            np.testing.assert_array_equal(reconstructed, image)
            offset += count

    @patch("forensics_app.tools.set7.codebook.extract_sift")
    def test_sift_is_applied_independently_to_every_pyramid_region(self, sift) -> None:
        sift.side_effect = lambda *_args: ([cv2.KeyPoint(2, 3, 4)], np.ones((1, 128), dtype=np.float32))
        settings = CodebookSettings(pyramid_levels=(1, 4, 16), max_image_size=64)
        gray, keypoints, descriptors = extract_pyramid_sift(Image.new("L", (128, 128)), settings)
        self.assertEqual(gray.shape, (64, 64))
        self.assertEqual(sift.call_count, 21)
        self.assertEqual([call.args[0].shape for call in sift.call_args_list], [(64, 64)] + [(32, 32)] * 4 + [(16, 16)] * 16)
        self.assertTrue(all(call.args[1] is settings for call in sift.call_args_list))
        self.assertEqual(len(descriptors), 21)
        self.assertEqual(keypoints[-1].pt, (50, 51))
        self.assertEqual([point.class_id for point in keypoints], list(range(21)))

    def test_histograms_use_nearest_words_and_preserve_region_order(self) -> None:
        centers = np.array([[0, 0], [10, 10]], dtype=np.float32)
        regions = [np.array([[1, 1], [9, 9], [8, 8]], dtype=np.float32), np.empty((0, 2))]
        np.testing.assert_array_equal(compute_histograms(regions, centers), [1, 2, 0, 0])

    def test_phow_dimensions_follow_pyramid_region_counts_and_k(self) -> None:
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (128, 128), dtype=np.uint8))
        centers = np.random.default_rng(42).random((10, 128)).astype(np.float32)
        for pyramid_levels, dimensions in (((1, 4, 16), 210), ((1, 4), 50), ((4,), 40)):
            with self.subTest(pyramid_levels=pyramid_levels):
                result = extract_phow(image, centers, CodebookSettings(pyramid_levels=pyramid_levels))
                self.assertEqual(result.shape, (dimensions,))
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.all(result >= 0))
                self.assertGreater(result.sum(), 0)

    def test_codebook_uses_only_training_images_and_has_ten_words(self) -> None:
        with TemporaryDirectory() as name:
            root = Path(name)
            random = np.random.default_rng(42)
            for label in CLASSES:
                (root / "train" / label).mkdir(parents=True)
                (root / "test" / label).mkdir(parents=True)
                for index in range(2):
                    pixels = random.integers(0, 256, (48, 48), dtype=np.uint8)
                    Image.fromarray(pixels).save(root / "train" / label / f"{index}.png")
                (root / "test" / label / "not_read.png").touch()
            model, patches, labels, paths, count, histograms = build_codebook(root / "train")
            self.assertEqual(model.cluster_centers_.shape, (10, 128))
            self.assertEqual(patches.shape, (10, 48, 48))
            self.assertEqual(len(paths), 6)
            self.assertEqual(set(labels), set(CLASSES))
            self.assertTrue(all(Path(path).parent.parent.name == "train" for path in paths))
            self.assertEqual(histograms.shape, (6, 210))
            self.assertEqual(int(histograms.sum()), count)
            self.assertGreater(count, 0)
            self.assertGreater(render_codebook(patches).width, 0)
            settings = CodebookSettings(words=3, step=4, sift_size=4, pyramid_levels=(1, 4), max_image_size=32)
            custom_model, custom_patches, _, _, custom_count, custom_histograms = build_codebook(root / "train", settings)
            self.assertEqual(custom_model.cluster_centers_.shape, (3, 128))
            self.assertEqual(custom_patches.shape, (3, 48, 48))
            self.assertEqual(custom_histograms.shape, (6, 15))
            self.assertEqual(int(custom_histograms.sum()), custom_count)

    @patch("forensics_app.tools.set7.codebook.filedialog.askdirectory", return_value="")
    def test_cancel_returns_none(self, _choose) -> None:
        self.assertIsNone(CodebookConstructionTool().run(None, ImageDocument()))

    @patch("forensics_app.tools.set7.codebook.choose_codebook_settings", return_value=CodebookSettings(
        words=3, step=4, sift_size=6, pyramid_levels=(1, 4), max_image_size=64,
        sigma_min=1.2, n_octaves=2, n_scales=4,
    ))
    @patch("forensics_app.tools.set7.codebook.build_codebook")
    @patch("forensics_app.tools.set7.codebook.filedialog.askdirectory", return_value="training")
    def test_tool_saves_centers_and_phow_settings(self, _choose, build, _settings) -> None:
        model = MagicMock()
        model.cluster_centers_ = np.ones((3, 128), dtype=np.float32)
        labels = np.array(CLASSES)
        paths = np.array([f"training/{label}/image.png" for label in CLASSES])
        build.return_value = model, np.zeros((3, 48, 48), dtype=np.uint8), labels, paths, 300, np.ones((3, 15))
        with TemporaryDirectory() as name:
            with patch("forensics_app.tools.set7.codebook.DEFAULT_DATA", Path(name)):
                result = CodebookConstructionTool().run(None, ImageDocument())
            build.assert_called_once_with(Path("training"), _settings.return_value)
            self.assertEqual(result.details["Visual words (K)"], 3)
            self.assertEqual(result.details["Training images"], 3)
            self.assertEqual(result.details["Descriptors"], 300)
            with np.load(result.details["Model"], allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["centers"], model.cluster_centers_)
                np.testing.assert_array_equal(archive["train_labels"], labels)
                np.testing.assert_array_equal(archive["train_paths"], paths)
                np.testing.assert_array_equal(archive["pyramid_levels"], [1, 4])
                self.assertEqual(archive["train_histograms"].shape, (3, 15))
                self.assertEqual(int(archive["sift_size"]), 6)
                self.assertEqual(int(archive["step"]), 4)
                self.assertEqual(int(archive["max_image_size"]), 64)
                self.assertEqual(float(archive["sigma_min"]), 1.2)
                self.assertEqual(int(archive["n_octaves"]), 2)
                self.assertEqual(int(archive["n_scales"]), 4)
                self.assertEqual(str(archive["descriptor_method"]), "spatial_pyramid_dense_sift_v3")

    def test_custom_sift_settings_control_sampling_resize_and_blur(self) -> None:
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (128, 128), dtype=np.uint8))
        settings = CodebookSettings(step=4, sift_size=4, pyramid_levels=(1,), max_image_size=32, sigma_min=1.2)
        gray, keypoints, regions = extract_pyramid_sift(image, settings)
        self.assertEqual(gray.shape, (32, 32))
        self.assertEqual(regions[0].shape, (len(keypoints), 128))
        _, _, other = extract_pyramid_sift(image, CodebookSettings(step=4, sift_size=4, pyramid_levels=(1,), max_image_size=32))
        self.assertFalse(np.array_equal(regions[0], other[0]))

    def test_dense_sift_uses_the_matching_gaussian_pyramid_layers(self) -> None:
        pixels = np.random.default_rng(42).integers(0, 256, (96, 96), dtype=np.uint8)
        keypoints, descriptors = extract_sift(pixels, CodebookSettings(n_octaves=2))
        self.assertTrue(any(point.octave == (1 | (1 << 8)) for point in keypoints))
        reference_points = [cv2.KeyPoint(*point.pt, point.size, 0, 0, point.octave) for point in keypoints]
        _, reference = cv2.SIFT_create(nOctaveLayers=3, sigma=1.6).compute(pixels, reference_points)
        np.testing.assert_array_equal(descriptors, reference)
        wrong_points = [cv2.KeyPoint(*point.pt, point.size, 0) for point in keypoints]
        _, wrong_layer = cv2.SIFT_create(nOctaveLayers=3, sigma=1.6).compute(pixels, wrong_points)
        self.assertFalse(np.array_equal(descriptors, wrong_layer))

    def test_small_regions_have_zero_histograms(self) -> None:
        keypoints, descriptors = extract_sift(np.zeros((8, 8), dtype=np.uint8), CodebookSettings())
        self.assertEqual(keypoints, [])
        self.assertEqual(descriptors.shape, (0, 128))
        histogram = extract_phow(Image.new("L", (8, 8)), np.zeros((10, 128)), CodebookSettings())
        np.testing.assert_array_equal(histogram, np.zeros(210))

    def test_scale_layers_control_dense_descriptors(self) -> None:
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (96, 96), dtype=np.uint8))
        first_keypoints, _ = extract_sift(np.array(image), CodebookSettings(n_scales=3))
        second_keypoints, _ = extract_sift(np.array(image), CodebookSettings(n_scales=5))
        self.assertEqual({(point.octave >> 8) & 255 for point in first_keypoints}, {0, 1, 2})
        self.assertEqual({(point.octave >> 8) & 255 for point in second_keypoints}, {0, 1, 2, 3, 4})

    def test_invalid_codebook_settings_are_rejected(self) -> None:
        for values in (
            {"words": 0}, {"step": 0}, {"sift_size": 0}, {"pyramid_levels": ()},
            {"pyramid_levels": (1, 3)}, {"pyramid_levels": (1, 1)}, {"pyramid_levels": (0,)},
            {"max_image_size": 0}, {"sigma_min": 0}, {"sigma_min": float("nan")},
            {"n_octaves": 0}, {"n_scales": 0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CodebookSettings(**values)

    @patch("forensics_app.tools.set7.codebook.build_codebook")
    @patch("forensics_app.tools.set7.codebook.choose_codebook_settings", return_value=None)
    @patch("forensics_app.tools.set7.codebook.filedialog.askdirectory", return_value="training")
    def test_cancel_settings_does_not_build(self, _folder, _settings, build) -> None:
        self.assertIsNone(CodebookConstructionTool().run(None, ImageDocument()))
        build.assert_not_called()

    def test_custom_word_counts_render_all_patches(self) -> None:
        for count in (1, 3, 12):
            with self.subTest(count=count):
                output = render_codebook(np.zeros((count, 48, 48), dtype=np.uint8))
                self.assertEqual(output.width, 200 * min(5, count))
                self.assertEqual(output.height, 225 * ((count + 4) // 5))


class VisualWordHistogramTests(unittest.TestCase):
    def test_load_codebook_preserves_extraction_settings(self) -> None:
        settings = CodebookSettings(words=3, step=4, sift_size=4, pyramid_levels=(1, 4), n_octaves=2, n_scales=2)
        centers = np.ones((3, 128), dtype=np.float32)
        with TemporaryDirectory() as name:
            path = Path(name) / "codebook.npz"
            np.savez_compressed(path, centers=centers, descriptor_method="spatial_pyramid_dense_sift_v3", **asdict(settings))
            model = load_codebook(path)
        self.assertEqual(model.settings, settings)
        np.testing.assert_array_equal(model.centers, centers)

    def test_load_codebook_rejects_old_or_corrupt_models(self) -> None:
        with TemporaryDirectory() as name:
            path = Path(name) / "codebook.npz"
            for values in (
                {"centers": np.ones((3, 128))},
                {"centers": np.ones((3, 128)), "descriptor_method": "opencv_dense_sift_bins_v2"},
                {"centers": np.ones((3, 12)), "descriptor_method": "spatial_pyramid_dense_sift_v3", **asdict(CodebookSettings())},
            ):
                np.savez_compressed(path, **values)
                with self.assertRaisesRegex(ValueError, "Rebuild"):
                    load_codebook(path)

    def test_codebook_rejects_nonfinite_centers(self) -> None:
        with self.assertRaises(ValueError):
            CodebookModel(np.full((10, 128), np.nan), CodebookSettings())

    def test_render_handles_empty_histograms(self) -> None:
        output = render_histogram(Image.new("L", (32, 32)), np.zeros(210), CodebookSettings())
        self.assertEqual(output.size, (1000, 600))

    @patch("forensics_app.tools.set7.histogram.choose_codebook", return_value="")
    def test_cancel_does_not_request_another_image(self, choose) -> None:
        document = MagicMock(current=Image.new("RGB", (64, 64)))
        self.assertIsNone(VisualWordHistogramTool().run(None, document))
        choose.assert_called_once_with(None)

    @patch("forensics_app.tools.set7.histogram.extract_phow", return_value=np.ones(210))
    @patch("forensics_app.tools.set7.histogram.load_codebook")
    @patch("forensics_app.tools.set7.histogram.choose_codebook", return_value="codebook.npz")
    def test_tool_uses_opened_image_and_returns_image_output(self, _choose, load, extract) -> None:
        model = CodebookModel(np.zeros((10, 128)), CodebookSettings())
        load.return_value = model
        image = Image.new("RGB", (64, 64))
        document = MagicMock(current=image)
        result = VisualWordHistogramTool().run(None, document)
        extract.assert_called_once_with(image, model.centers, model.settings)
        self.assertIsInstance(result.image, Image.Image)
        self.assertEqual(result.image.size, (1000, 600))
        self.assertIsNot(result.image, image)
        self.assertIs(document.current, image)
        self.assertEqual(result.details["PHOW dimensions"], 210)


class VisualWordClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = CodebookSettings(words=3, pyramid_levels=(1, 4), sift_size=4, max_image_size=64)
        self.codebook = CodebookModel(np.zeros((3, 128), dtype=np.float32), self.settings)

    def make_training_images(self, root: Path) -> None:
        for index, label in enumerate(CLASSES):
            folder = root / "train" / label
            folder.mkdir(parents=True)
            for number in range(4):
                Image.new("L", (48, 48), index * 80).save(folder / f"{number}.png")
            test_folder = root / "test" / label
            test_folder.mkdir(parents=True)
            (test_folder / "not_read.png").touch()

    def encoded_image(self, image, _centers, _settings) -> np.ndarray:
        histogram = np.zeros(15, dtype=np.float32)
        histogram[image.getpixel((0, 0)) // 80] = 100
        return histogram

    @patch("forensics_app.tools.set7.classification.extract_phow")
    def test_training_and_save_load_round_trip_are_self_contained(self, extract) -> None:
        extract.side_effect = self.encoded_image
        image = Image.new("L", (48, 48), 80)
        with TemporaryDirectory() as name:
            root = Path(name)
            self.make_training_images(root)
            model = train_classifier(root / "train", self.codebook)
            self.assertEqual(model.training_images, 12)
            self.assertEqual(set(model.classifier.classes_), set(CLASSES))
            self.assertEqual(model.classifier.named_steps["normalizer"].norm, "l1")
            self.assertTrue(model.classifier.named_steps["svc"].probability)
            self.assertEqual(extract.call_count, 12)
            before = predict_image(image, model)
            path = root / "classifier.joblib"
            save_classifier(path, model)
            loaded = load_classifier(path)
            after = predict_image(image, loaded)
            mismatched_codebook = CodebookModel(self.codebook.centers, CodebookSettings(words=3, pyramid_levels=(1,)))
            save_classifier(path, VisualWordClassifier(mismatched_codebook, model.classifier, 12))
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_classifier(path)
        self.assertEqual(loaded.codebook.settings, self.settings)
        np.testing.assert_array_equal(loaded.codebook.centers, self.codebook.centers)
        self.assertEqual(before[:3], after[:3])
        np.testing.assert_array_equal(before[3], after[3])
        label, probability, scores, _ = after
        self.assertEqual(label, max(scores, key=scores.get))
        self.assertEqual(probability, max(scores.values()))
        self.assertAlmostEqual(sum(scores.values()), 1)

    def test_missing_training_class_is_rejected(self) -> None:
        with TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, "Missing training class"):
                train_classifier(Path(name), self.codebook)

    def test_too_few_training_images_are_rejected(self) -> None:
        with TemporaryDirectory() as name:
            folder = Path(name) / CLASSES[0]
            folder.mkdir()
            Image.new("L", (48, 48)).save(folder / "one.png")
            with self.assertRaisesRegex(ValueError, "At least two"):
                train_classifier(Path(name), self.codebook)

    @patch("forensics_app.tools.set7.classification.extract_phow", return_value=np.zeros(15))
    def test_featureless_training_and_query_images_are_rejected(self, _extract) -> None:
        with TemporaryDirectory() as name:
            self.make_training_images(Path(name))
            with self.assertRaisesRegex(ValueError, "No SIFT descriptors"):
                train_classifier(Path(name) / "train", self.codebook)
        model = VisualWordClassifier(self.codebook, MagicMock(), 12)
        with self.assertRaisesRegex(ValueError, "No SIFT descriptors"):
            predict_image(Image.new("L", (8, 8)), model)

    @patch("forensics_app.tools.set7.classification.extract_phow", return_value=np.ones(15))
    def test_prediction_selects_highest_probability_not_svm_decision(self, _extract) -> None:
        classifier = MagicMock()
        classifier.classes_ = np.array(CLASSES)
        classifier.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        model = VisualWordClassifier(self.codebook, classifier, 12)
        label, probability, scores, histogram = predict_image(Image.new("L", (48, 48)), model)
        self.assertEqual(label, "kangaroo")
        self.assertEqual(probability, 0.7)
        self.assertEqual(scores["kangaroo"], 0.7)
        self.assertEqual(histogram.shape, (15,))
        classifier.predict.assert_not_called()

    @patch("forensics_app.tools.set7.classification.joblib.load", return_value={"format": "wrong"})
    def test_incompatible_classifier_is_rejected(self, _load) -> None:
        with self.assertRaisesRegex(ValueError, "Not a supported"):
            load_classifier(Path("classifier.joblib"))

    @patch("forensics_app.tools.set7.classification.choose_option", return_value=None)
    def test_cancel_mode_returns_none(self, _choose) -> None:
        self.assertIsNone(VisualWordPredictionTool().run(None, MagicMock(current=Image.new("L", (48, 48)))))

    @patch("forensics_app.tools.set7.classification.predict_image")
    @patch("forensics_app.tools.set7.classification.load_classifier")
    @patch("forensics_app.tools.set7.classification.filedialog.askopenfilename", return_value="classifier.joblib")
    @patch("forensics_app.tools.set7.classification.choose_option", return_value="load")
    def test_saved_classifier_returns_prediction_image_without_training(self, _mode, _file, load, predict) -> None:
        model = VisualWordClassifier(self.codebook, MagicMock(), 12)
        load.return_value = model
        predict.return_value = "kangaroo", 0.7, dict(zip(CLASSES, (0.1, 0.2, 0.7))), np.ones(15)
        image = Image.new("L", (48, 48))
        result = VisualWordPredictionTool().run(None, MagicMock(current=image))
        load.assert_called_once_with(Path("classifier.joblib"))
        predict.assert_called_once_with(image, model)
        self.assertEqual(result.details["Prediction"], "kangaroo")
        self.assertEqual(result.details["Top probability"], 0.7)
        self.assertIsInstance(result.image, Image.Image)
        self.assertEqual(result.image.size, (1000, 600))
        self.assertIsNot(result.image, image)

    @patch("forensics_app.tools.set7.classification.load_classifier")
    @patch("forensics_app.tools.set7.classification.filedialog.askopenfilename", return_value="")
    @patch("forensics_app.tools.set7.classification.choose_option", return_value="load")
    def test_cancel_classifier_picker_does_not_load(self, _mode, _file, load) -> None:
        self.assertIsNone(VisualWordPredictionTool().run(None, MagicMock(current=Image.new("L", (48, 48)))))
        load.assert_not_called()

    @patch("forensics_app.tools.set7.classification.predict_image")
    @patch("forensics_app.tools.set7.classification.save_classifier")
    @patch("forensics_app.tools.set7.classification.train_classifier")
    @patch("forensics_app.tools.set7.classification.filedialog.asksaveasfilename", return_value="new.joblib")
    @patch("forensics_app.tools.set7.classification.filedialog.askdirectory", return_value="selected_training")
    @patch("forensics_app.tools.set7.classification.load_codebook")
    @patch("forensics_app.tools.set7.classification.choose_codebook", return_value="codebook.npz")
    @patch("forensics_app.tools.set7.classification.choose_option", return_value="train")
    def test_train_mode_uses_selected_folder_saves_then_predicts(
        self, _mode, _file, load, _folder, _save_file, train, save, predict,
    ) -> None:
        load.return_value = self.codebook
        model = VisualWordClassifier(self.codebook, MagicMock(), 12)
        train.return_value = model
        predict.return_value = "butterfly", 0.8, dict(zip(CLASSES, (0.8, 0.1, 0.1))), np.ones(15)
        image = Image.new("L", (48, 48))
        result = VisualWordPredictionTool().run(None, MagicMock(current=image))
        train.assert_called_once_with(Path("selected_training"), self.codebook)
        save.assert_called_once_with(Path("new.joblib"), model)
        predict.assert_called_once_with(image, model)
        self.assertIn("Saved classifier", result.message)
        self.assertIsInstance(result.image, Image.Image)
        self.assertEqual(result.image.size, (1000, 600))

    def test_cancel_training_inputs_does_not_train_or_save(self) -> None:
        for codebook_name, folder_name, save_name in (
            ("", "training", "new.joblib"),
            ("codebook.npz", "", "new.joblib"),
            ("codebook.npz", "training", ""),
        ):
            with (
                self.subTest(codebook=codebook_name, folder=folder_name, save=save_name),
                patch("forensics_app.tools.set7.classification.choose_option", return_value="train"),
                patch("forensics_app.tools.set7.classification.choose_codebook", return_value=codebook_name),
                patch("forensics_app.tools.set7.classification.load_codebook", return_value=self.codebook),
                patch("forensics_app.tools.set7.classification.filedialog.askdirectory", return_value=folder_name),
                patch("forensics_app.tools.set7.classification.filedialog.asksaveasfilename", return_value=save_name),
                patch("forensics_app.tools.set7.classification.train_classifier") as train,
                patch("forensics_app.tools.set7.classification.save_classifier") as save,
            ):
                self.assertIsNone(VisualWordPredictionTool().run(None, MagicMock(current=Image.new("L", (48, 48)))))
                train.assert_not_called()
                save.assert_not_called()


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


class FaceDetectionTests(unittest.TestCase):
    def test_face_detection_is_registered_in_set6(self) -> None:
        registry = build_tool_registry()
        tools = dict(registry.categories())["Set 6"]
        self.assertTrue(any(isinstance(tool, FaceDetectionTool) for tool in tools))

    @patch("forensics_app.tools.set6.face_detection.cv2.CascadeClassifier")
    def test_haar_uses_grayscale_and_selected_settings(self, cascade) -> None:
        classifier = cascade.return_value
        classifier.empty.return_value = False
        classifier.detectMultiScale.return_value = np.array([[5, 6, 12, 14]])
        image = Image.new("RGBA", (40, 40), (255, 0, 0, 100))

        output, count = detect_faces(image, FaceDetectionSettings("haar", 1.2, 4, 12))

        self.assertEqual(count, 1)
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, image.size)
        self.assertEqual(classifier.detectMultiScale.call_args.args[0].shape, (40, 40))
        self.assertEqual(classifier.detectMultiScale.call_args.kwargs, {
            "scaleFactor": 1.2, "minNeighbors": 4, "minSize": (12, 12),
        })
        self.assertEqual(output.getpixel((5, 6)), (0, 255, 0))
        self.assertEqual(image.getpixel((5, 6)), (255, 0, 0, 100))

    @patch("forensics_app.tools.set6.face_detection.Path.is_file", return_value=True)
    @patch("forensics_app.tools.set6.face_detection.cv2.FaceDetectorYN.create")
    def test_yunet_uses_bgr_and_draws_landmarks(self, create, _is_file) -> None:
        create.return_value.detect.return_value = (1, np.array([
            [5, 25, 30, 30, 12, 35, 28, 35, 20, 40, 14, 48, 26, 48, 0.95],
        ]))
        settings = FaceDetectionSettings("yunet", score_threshold=0.8, nms_threshold=0.4, model_path="face.onnx")

        output, count = detect_faces(Image.new("RGB", (60, 70), "red"), settings)

        create.assert_called_once_with("face.onnx", "", (60, 70), 0.8, 0.4, 5000)
        np.testing.assert_array_equal(create.return_value.detect.call_args.args[0][0, 0], [0, 0, 255])
        self.assertEqual(count, 1)
        self.assertEqual(output.size, (60, 70))
        self.assertEqual(output.getpixel((12, 35)), (0, 100, 255))

    @patch("forensics_app.tools.set6.face_detection.Path.is_file", return_value=True)
    @patch("forensics_app.tools.set6.face_detection.cv2.FaceDetectorYN.create")
    def test_yunet_no_faces_preserves_image(self, create, _is_file) -> None:
        create.return_value.detect.return_value = (0, None)
        image = Image.new("RGB", (40, 40), "red")

        output, count = detect_faces(image, FaceDetectionSettings("yunet"))

        self.assertEqual(count, 0)
        np.testing.assert_array_equal(np.array(output), np.array(image))

    @patch("forensics_app.tools.set6.face_detection.cv2.CascadeClassifier")
    def test_haar_empty_detections_and_missing_cascade(self, cascade) -> None:
        cascade.return_value.empty.return_value = False
        cascade.return_value.detectMultiScale.return_value = ()
        image = Image.new("L", (40, 40))
        output, count = detect_faces(image, FaceDetectionSettings("haar"))
        self.assertEqual(count, 0)
        self.assertEqual(output.mode, "RGB")
        cascade.return_value.empty.return_value = True
        with self.assertRaisesRegex(ValueError, "cascade"):
            detect_faces(image, FaceDetectionSettings("haar"))

    def test_invalid_settings_are_rejected(self) -> None:
        image = Image.new("RGB", (40, 40))
        for settings in (
            FaceDetectionSettings("unknown"),
            FaceDetectionSettings("haar", scale_factor=1),
            FaceDetectionSettings("haar", scale_factor=float("nan")),
            FaceDetectionSettings("haar", min_neighbors=-1),
            FaceDetectionSettings("haar", min_size=0),
            FaceDetectionSettings("yunet", score_threshold=0),
            FaceDetectionSettings("yunet", nms_threshold=float("nan")),
            FaceDetectionSettings("yunet", model_path="missing.onnx"),
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                detect_faces(image, settings)

    @patch("forensics_app.tools.set6.face_detection.choose_face_detection", return_value=None)
    def test_cancel_returns_none(self, _choose) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (40, 40))
        self.assertIsNone(FaceDetectionTool().run(None, document))

    @patch("forensics_app.tools.set6.face_detection.choose_face_detection", return_value=FaceDetectionSettings("haar"))
    @patch("forensics_app.tools.set6.face_detection.detect_faces")
    def test_tool_returns_image_and_detection_details(self, detect, _choose) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (40, 40))
        detect.return_value = document.current.copy(), 2
        result = FaceDetectionTool().run(None, document)
        self.assertEqual(result.details["Faces detected"], 2)
        self.assertEqual(result.details["Method"], "Haar / AdaBoost")
        self.assertEqual(result.image.size, (40, 40))


class ToolTests(unittest.TestCase):
    @patch(
        "forensics_app.tools.set4.person_detection.choose_person_detection",
        return_value=PersonDetectionSettings(9, 8, 3, "dot_product"),
    )
    @patch(
        "forensics_app.tools.set4.person_detection.filedialog.askopenfilename",
        return_value="person.png",
    )
    @patch("forensics_app.tools.set4.person_detection.Image.open")
    @patch("forensics_app.tools.set4.person_detection.feature.hog")
    def test_person_detection_uses_selected_hog_settings_and_metric(
        self, hog, open_image, _choose_file, _choose_settings
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (24, 24), "white")
        template_context = MagicMock()
        template_context.__enter__.return_value = Image.new("RGB", (16, 16), "white")
        open_image.return_value = template_context
        hog.side_effect = [
            np.array([1.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([3.0, 0.0]),
            np.array([4.0, 0.0]),
        ]

        result = PersonDetectionTool().run(None, document)

        self.assertEqual(hog.call_count, 5)
        self.assertEqual(hog.call_args.kwargs, {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (3, 3),
            "channel_axis": -1,
        })
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, (24, 24))
        self.assertEqual(result.details["Metric"], "Dot product")
        self.assertEqual(result.details["Location"], "8, 8")

    @patch(
        "forensics_app.tools.set4.hog.choose_hog",
        return_value=HOGSettings(9, 8, 3),
    )
    @patch("forensics_app.tools.set4.hog.feature.hog")
    def test_hog_uses_selected_configuration_and_returns_visualization(
        self, hog, _choose_hog
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("RGBA", (16, 16), (128, 128, 128, 64))
        hog.return_value = np.ones(36), np.ones((16, 16))

        result = HOGTool().run(None, document)

        self.assertEqual(hog.call_args.kwargs, {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (3, 3),
            "visualize": True,
        })
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, document.current.size)
        self.assertEqual(result.details["Orientations"], 9)

    @patch(
        "forensics_app.tools.set4.template_matching.choose_template_matching",
        return_value=TemplateMatchingSettings(45.0, 0.8),
    )
    @patch(
        "forensics_app.tools.set4.template_matching.filedialog.askopenfilename",
        return_value="template.png",
    )
    @patch("forensics_app.tools.set4.template_matching.Image.open")
    @patch("forensics_app.tools.set4.template_matching.transform.rotate")
    @patch("forensics_app.tools.set4.template_matching.feature.match_template")
    def test_template_matching_rotates_template_and_uses_selected_threshold(
        self, match_template, rotate, open_image, _choose_file, _choose_settings
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("RGBA", (5, 5), (128, 128, 128, 64))
        template_context = MagicMock()
        template_context.__enter__.return_value = Image.new("RGBA", (2, 2), (255, 255, 255, 64))
        open_image.return_value = template_context
        rotate.return_value = np.ones((2, 2))
        response = np.zeros((5, 5))
        response[2, 3] = 0.9
        match_template.return_value = response

        result = TemplateMatchingTool().run(None, document)

        rotate.assert_called_once_with(rotate.call_args.args[0], 45.0)
        self.assertEqual(match_template.call_args.kwargs, {"pad_input": True})
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.details["Angle"], 45.0)
        self.assertEqual(result.details["Threshold"], 0.8)

    @patch(
        "forensics_app.tools.set4.normalized_cross_correlation.filedialog.askopenfilename",
        return_value="template.png",
    )
    @patch("forensics_app.tools.set4.normalized_cross_correlation.Image.open")
    @patch("forensics_app.tools.set4.normalized_cross_correlation.feature.match_template")
    def test_normalized_cross_correlation_returns_response_map(
        self, match_template, open_image, _choose_file
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("RGBA", (4, 3), (128, 128, 128, 64))
        template_context = MagicMock()
        template_context.__enter__.return_value = Image.new("RGBA", (2, 2), (255, 255, 255, 64))
        open_image.return_value = template_context
        match_template.return_value = np.array(
            [[-1.0, -0.5, 0.0, 0.5], [0.0, 0.8, 1.0, 0.0], [-1.0, 0.0, 0.2, 0.0]]
        )

        result = NormalizedCrossCorrelationTool().run(None, document)

        self.assertEqual(match_template.call_args.kwargs, {"pad_input": True})
        self.assertEqual(match_template.call_args.args[0].shape, (3, 4))
        self.assertEqual(match_template.call_args.args[1].shape, (2, 2))
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, document.current.size)
        self.assertEqual(result.details["Operation"], "Normalized cross-correlation image")
        self.assertNotIn("Threshold", result.details)

    @patch(
        "forensics_app.tools.set3.image_sharpening.choose_sharpening",
        return_value=SharpeningSettings(2.0, 1.5),
    )
    @patch("forensics_app.tools.set3.image_sharpening.filters.unsharp_mask")
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
        })
        self.assertEqual(list(result.image.getdata()), [0, 255])
        self.assertEqual(result.details["Radius"], 2.0)
        self.assertEqual(result.details["Amount"], 1.5)

    @patch(
        "forensics_app.tools.set3.edge_visualization.choose_edge_visualization",
        return_value=EdgeVisualizationSettings(2.0, 0.1, 0.2, 2.0),
    )
    @patch(
        "forensics_app.tools.set3.edge_visualization.feature.canny",
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
        "forensics_app.tools.set3.canny_edge_detection.choose_canny",
        return_value=CannySettings(2.0, 0.1, 0.2),
    )
    @patch(
        "forensics_app.tools.set3.canny_edge_detection.feature.canny",
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
                    "forensics_app.tools.set3.filters.choose_filter",
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
            "forensics_app.tools.set3.filters.choose_filter",
            return_value=FilterSettings("gaussian", 0.5),
        ):
            low_sigma = FiltersTool().run(None, document)
        with patch(
            "forensics_app.tools.set3.filters.choose_filter",
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
                    "forensics_app.tools.set3.convolution.choose_convolution",
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

    @patch("forensics_app.tools.set2.channel_split.choose_channel", return_value="Green")
    def test_channel_split_extracts_selected_channel_without_mutating_document(self, _choose_channel) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1), (12, 128, 240))

        result = ChannelSplitTool().run(None, document)

        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.getpixel((0, 0)), 128)
        self.assertEqual(result.details["Channel"], "Green")
        self.assertEqual(document.current.mode, "RGB")

    @patch("forensics_app.tools.set2.channel_split.choose_channel", return_value=None)
    def test_channel_split_returns_none_when_cancelled(self, _choose_channel) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1), "red")

        self.assertIsNone(ChannelSplitTool().run(None, document))

    @patch("forensics_app.tools.set2.channel_split.choose_channel", return_value="Blue")
    def test_channel_split_accepts_grayscale_image(self, _choose_channel) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (1, 1), 73)

        result = ChannelSplitTool().run(None, document)

        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.getpixel((0, 0)), 73)

    @patch(
        "forensics_app.tools.set2.channel_swap.choose_channel",
        side_effect=["Red", "Blue"],
    )
    def test_channel_swap_swaps_selected_rgb_channels(self, _askstring) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1), (10, 20, 30))

        result = ChannelSwapTool().run(None, document)

        self.assertEqual(result.image.getpixel((0, 0)), (30, 20, 10))
        self.assertEqual(result.details["Channels"], "Red and Blue")
        self.assertEqual(document.current.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(_askstring.call_args.kwargs["excluded_channels"], {"Red"})

    @patch(
        "forensics_app.tools.set2.channel_swap.choose_channel",
        side_effect=["Green", "Red"],
    )
    def test_channel_swap_preserves_alpha(self, _askstring) -> None:
        document = ImageDocument()
        document.current = Image.new("RGBA", (1, 1), (10, 20, 30, 40))

        result = ChannelSwapTool().run(None, document)

        self.assertEqual(result.image.getpixel((0, 0)), (20, 10, 30, 40))

    @patch("forensics_app.tools.set2.channel_swap.choose_channel", side_effect=["Red", "Blue"])
    def test_channel_swap_accepts_grayscale_image(self, _choose_channel) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (1, 1), 73)

        result = ChannelSwapTool().run(None, document)

        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.getpixel((0, 0)), (73, 73, 73))

    @patch(
        "forensics_app.tools.set2.masking.choose_masking",
        return_value=MaskingSettings(True, True, False, 135),
    )
    @patch("forensics_app.tools.set2.masking.filedialog.askopenfilename", return_value="foreground.png")
    @patch("forensics_app.tools.set2.masking.Image.open")
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
        "forensics_app.tools.set2.masking.choose_masking",
        return_value=MaskingSettings(False, True, False, 135),
    )
    @patch("forensics_app.tools.set2.masking.filedialog.askopenfilename", return_value="foreground.png")
    @patch("forensics_app.tools.set2.masking.Image.open")
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
        "forensics_app.tools.set2.masking.choose_masking",
        return_value=MaskingSettings(False, True, False, 135),
    )
    @patch("forensics_app.tools.set2.masking.filedialog.askopenfilename", return_value="foreground.png")
    @patch("forensics_app.tools.set2.masking.Image.open")
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
        "forensics_app.tools.set2.masking.choose_masking",
        return_value=MaskingSettings(False, False, False, 135),
    )
    @patch("forensics_app.tools.set2.masking.filedialog.askopenfilename", return_value="foreground.png")
    @patch("forensics_app.tools.set2.masking.Image.open")
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
        "forensics_app.tools.set2.masking.choose_masking",
        return_value=MaskingSettings(False, True, False, 200),
    )
    @patch("forensics_app.tools.set2.masking.filedialog.askopenfilename", return_value="foreground.png")
    @patch("forensics_app.tools.set2.masking.Image.open")
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
        "forensics_app.tools.set2.masking.choose_masking",
        return_value=MaskingSettings(False, True, False, 135, True),
    )
    @patch("forensics_app.tools.set2.masking.filedialog.askopenfilename", return_value="foreground.png")
    @patch("forensics_app.tools.set2.masking.Image.open")
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

    def test_registry_groups_tools_into_course_sets(self) -> None:
        registry = build_tool_registry()

        self.assertEqual(
            tuple((category, tuple(tool.tool_id for tool in tools)) for category, tools in registry.categories()),
            (
                ("Set 7", ("codebook_construction", "visual_word_histogram", "visual_word_prediction")),
                ("Set 6", ("face_detection", "eigenface_builder", "face_recognition")),
                ("Set 5", ("corner_detection", "keypoint_descriptors", "image_matching", "closest_image_lookup")),
                ("Set 4", ("normalized_cross_correlation", "template_matching", "hog", "person_detection")),
                ("Set 3", ("canny_edge_detection", "convolution", "filters", "edge_visualization", "image_sharpening")),
                ("Set 2", ("channel_split", "channel_swap", "masking", "histogram", "contrast_enhancement")),
                ("Set 1", ("grayscale", "image_info")),
            ),
        )

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
                    "forensics_app.tools.set2.contrast_enhancement.choose_option",
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
                    "forensics_app.tools.set2.contrast_enhancement.choose_option",
                    return_value=method,
                ):
                    result = ContrastEnhancementTool().run(None, document)

                pixel = result.image.getpixel((8, 8))
                self.assertEqual(result.image.mode, "RGB")
                self.assertGreater(len(set(pixel)), 1)
                self.assertEqual(document.current.getpixel((8, 8)), (136, 119, 132))

    @patch("forensics_app.tools.set5.utils.feature.ORB")
    def test_feature_extraction_accepts_grayscale_and_rgba_images(self, create_descriptor) -> None:
        grayscale = Image.new("L", (2, 1), 128)
        rgba = Image.new("RGBA", (2, 1), (255, 0, 0, 50))
        descriptor = MagicMock()
        descriptor.keypoints = np.empty((0, 2))
        descriptor.descriptors = np.empty((0, 256))
        create_descriptor.return_value = descriptor
        settings = DescriptorSettings("orb", 200, 3, 3, 1.6)

        extract_features(grayscale, settings)
        extract_features(rgba, settings)

        grayscale_array = descriptor.detect_and_extract.call_args_list[0].args[0]
        rgba_array = descriptor.detect_and_extract.call_args_list[1].args[0]
        self.assertEqual(grayscale_array.shape, (1, 2))
        self.assertEqual(rgba_array.shape, (1, 2))
        self.assertEqual(grayscale_array.tolist(), [[128, 128]])

    @patch("forensics_app.tools.set5.utils.feature.match_descriptors")
    def test_feature_matching_selects_metric_for_descriptor_algorithm(self, match_descriptors) -> None:
        descriptors = np.array([[0, 1], [1, 0]])
        matching = MatchingSettings(True, 0.8)

        match_features(descriptors, descriptors, DescriptorSettings("orb", 200, 3, 3, 1.6), matching)
        self.assertEqual(match_descriptors.call_args.kwargs["metric"], "hamming")

        match_features(descriptors, descriptors, DescriptorSettings("sift", 200, 3, 3, 1.6), matching)
        self.assertEqual(match_descriptors.call_args.kwargs["metric"], "euclidean")

    @patch(
        "forensics_app.tools.set5.corner_detection.choose_corner_detection",
        return_value=CornerSettings("harris", 0.25),
    )
    @patch("forensics_app.tools.set5.corner_detection.feature.corner_peaks", return_value=np.array([[1, 2]]))
    @patch("forensics_app.tools.set5.corner_detection.feature.corner_harris", return_value=np.ones((3, 4)))
    def test_corner_detection_uses_selected_detector_and_peak_threshold(
        self, corner_harris, corner_peaks, _choose_corner_detection
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (4, 3), 128)

        result = CornerDetectionTool().run(None, document)

        self.assertEqual(corner_harris.call_args.args[0].shape, (3, 4))
        self.assertEqual(corner_peaks.call_args.kwargs, {"threshold_rel": 0.25})
        self.assertEqual(result.details["Corners"], 1)
        self.assertEqual(result.image.mode, "RGB")

    @patch(
        "forensics_app.tools.set5.corner_detection.choose_corner_detection",
        return_value=CornerSettings("fast", 0.25, "response"),
    )
    @patch("forensics_app.tools.set5.corner_detection.render_response", return_value=Image.new("RGB", (4, 3)))
    @patch("forensics_app.tools.set5.corner_detection.feature.corner_peaks")
    @patch("forensics_app.tools.set5.corner_detection.feature.corner_fast", return_value=np.ones((3, 4)))
    def test_corner_detection_can_return_raw_detector_response(
        self, corner_fast, corner_peaks, _render_response, _choose_corner_detection
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "white")

        result = CornerDetectionTool().run(None, document)

        self.assertEqual(corner_fast.call_args.args[0].shape, (3, 4))
        corner_peaks.assert_not_called()
        self.assertEqual(result.details["Operation"], "FAST corner response")

    @patch("forensics_app.tools.set5.image_matching.render_matches", return_value=Image.new("RGB", (4, 2)))
    @patch("forensics_app.tools.set5.image_matching.match_features", return_value=np.array([[0, 0]]))
    @patch(
        "forensics_app.tools.set5.image_matching.extract_features",
        side_effect=[(np.array([[0, 0]]), np.array([[1, 0]])), (np.array([[0, 0]]), np.array([[1, 0]]))],
    )
    @patch(
        "forensics_app.tools.set5.image_matching.choose_matching_settings",
        return_value=MatchingSettings(True, 0.8),
    )
    @patch(
        "forensics_app.tools.set5.image_matching.choose_descriptor_settings",
        return_value=DescriptorSettings("orb", 200, 3, 3, 1.6),
    )
    @patch("forensics_app.tools.set5.image_matching.filedialog.askopenfilename", return_value="second.png")
    @patch("forensics_app.tools.set5.image_matching.Image.open")
    def test_image_matching_reports_second_image_match_count(
        self, open_image, _choose_file, _choose_descriptor, _choose_matching, _extract, _match, _render
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (2, 2), "white")
        source = MagicMock()
        source.__enter__.return_value = Image.new("L", (2, 2), 128)
        open_image.return_value = source

        result = ImageMatchingTool().run(None, document)

        self.assertEqual(result.details["Second image"], "second.png")
        self.assertEqual(result.details["Matches"], 1)

    @patch("forensics_app.tools.set5.closest_image_lookup.render_matches", return_value=Image.new("RGB", (4, 2)))
    @patch(
        "forensics_app.tools.set5.closest_image_lookup.match_features",
        side_effect=[np.array([[0, 0]]), np.array([[0, 0], [1, 1]])],
    )
    @patch(
        "forensics_app.tools.set5.closest_image_lookup.extract_features",
        side_effect=[
            (np.array([[0, 0]]), np.array([[1, 0]])),
            (np.array([[0, 0]]), np.array([[1, 0]])),
            (np.array([[0, 0]]), np.array([[1, 0]])),
        ],
    )
    @patch(
        "forensics_app.tools.set5.closest_image_lookup.choose_matching_settings",
        return_value=MatchingSettings(True, 0.8),
    )
    @patch(
        "forensics_app.tools.set5.closest_image_lookup.choose_descriptor_settings",
        return_value=DescriptorSettings("orb", 200, 3, 3, 1.6),
    )
    def test_closest_image_lookup_selects_candidate_with_most_matches(
        self, _choose_descriptor, _choose_matching, _extract, _match, _render
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (2, 2), "white")
        with TemporaryDirectory() as directory:
            Image.new("RGB", (2, 2), "red").save(f"{directory}/first.png")
            Image.new("RGB", (2, 2), "blue").save(f"{directory}/second.png")
            with patch("forensics_app.tools.set5.closest_image_lookup.filedialog.askdirectory", return_value=directory):
                result = ClosestImageLookupTool().run(None, document)

        self.assertEqual(result.details["Closest image"], "second.png")
        self.assertEqual(result.details["Matches"], 2)
        self.assertEqual(result.details["Candidates examined"], 2)


if __name__ == "__main__":
    unittest.main()
