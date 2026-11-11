import unittest
import json
import importlib.util
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
    CLASSES, CodebookConstructionTool, CodebookSettings, assign_features_to_words, build_codebook, compute_histograms,
    CodebookModel, extract_phow, extract_pyramid_sift, extract_sift, load_codebook, render_codebook, split_pyramid,
)
from forensics_app.tools.set7.histogram import VisualWordHistogramTool, render_histogram
from forensics_app.tools.set7.classification import (
    VisualWordClassifier, VisualWordPredictionTool, load_classifier, predict_image, save_classifier, train_classifier,
)
from forensics_app.tools.set8.classification import (
    CNNSettings, CNNTrainingTool, build_cnn, describe_model, get_keras, load_training_images, prepare_image,
    render_training_history, save_cnn, train_cnn,
)
from forensics_app.tools.set8.classification import log_progress, select_sample_indices
from forensics_app.tools.set8.prediction import CNNPredictionTool, load_cnn, passport_details
from forensics_app.tools.set8.prediction import predict_image as predict_cnn_image
from forensics_app.tools.set8.features import CNNFeatureExtractionTool, FeatureVectors, load_features, save_features
from forensics_app.tools.set8.features import extract_features as extract_cnn_features
from forensics_app.tools.set8.projection import CNNFeatureProjectionTool, ProjectionSettings, project_features, render_projection
from forensics_app.tools.utils import (
    add_dialog_buttons,
    add_numeric_field,
    choose_option,
    create_dialog,
    wait_for_dialog,
)


class CNNProjectionTests(unittest.TestCase):
    def vectors(self) -> FeatureVectors:
        return FeatureVectors(np.random.default_rng(42).normal(size=(12, 8)).astype(np.float32),
                             np.array([0] * 6 + [1] * 6), np.array([str(index) for index in range(12)]), ("cat", "dog"), {})

    def test_tsne_projects_to_finite_2d_and_renders(self) -> None:
        vectors = self.vectors()
        coordinates = project_features(vectors, ProjectionSettings(perplexity=3, iterations=250))
        self.assertEqual(coordinates.shape, (12, 2))
        self.assertTrue(np.isfinite(coordinates).all())
        image = render_projection(coordinates, vectors, "tsne")
        self.assertEqual(image.mode, "RGB")
        self.assertGreater(image.width, 0)

    def test_projection_rejects_invalid_parameters(self) -> None:
        for parameters in ({"method": "bad"}, {"perplexity": 0}, {"iterations": 10}, {"n_neighbors": 1}, {"min_dist": 2}):
            with self.assertRaises(ValueError):
                ProjectionSettings(**parameters)
        with self.assertRaisesRegex(ValueError, "perplexity"):
            project_features(self.vectors())
        with self.assertRaisesRegex(ValueError, "neighbors"):
            project_features(self.vectors(), ProjectionSettings(method="umap", n_neighbors=20))

    @unittest.skipUnless(importlib.util.find_spec("umap"), "UMAP is optional")
    def test_umap_projects_to_finite_2d(self) -> None:
        coordinates = project_features(self.vectors(), ProjectionSettings(method="umap", n_neighbors=3))
        self.assertEqual(coordinates.shape, (12, 2))
        self.assertTrue(np.isfinite(coordinates).all())

    @patch("forensics_app.tools.set8.projection.filedialog.askopenfilename", return_value="")
    def test_cancel_returns_none_without_an_open_image(self, _choose) -> None:
        self.assertFalse(CNNFeatureProjectionTool.requires_image)
        self.assertIsNone(CNNFeatureProjectionTool().run(None, ImageDocument()))

    def test_tool_loads_saved_features_without_model_or_original_images(self) -> None:
        with TemporaryDirectory() as name:
            path = Path(name) / "features.npz"
            save_features(path, self.vectors())
            with patch("forensics_app.tools.set8.projection.filedialog.askopenfilename", return_value=str(path)), patch(
                "forensics_app.tools.set8.projection.choose_projection_settings", return_value=ProjectionSettings(perplexity=3, iterations=250)
            ):
                result = CNNFeatureProjectionTool().run(None, ImageDocument())
            self.assertEqual(result.details["Samples"], 12)
            self.assertEqual(result.details["Feature dimensions"], 8)
            self.assertEqual(result.details["method"], "tsne")
            self.assertEqual(result.image.mode, "RGB")


class CNNFeatureTests(unittest.TestCase):
    def test_feature_archive_round_trip_and_invalid_labels(self) -> None:
        vectors = FeatureVectors(np.ones((4, 3), dtype=np.float32), np.array([0, 0, 1, 1]),
                                 np.array(["a", "b", "c", "d"]), ("cat", "dog"), {"feature_layer": "features"})
        with TemporaryDirectory() as name:
            path = Path(name) / "features.npz"
            save_features(path, vectors)
            loaded = load_features(path)
            np.testing.assert_array_equal(loaded.features, vectors.features)
            np.testing.assert_array_equal(loaded.labels, vectors.labels)
            np.testing.assert_array_equal(loaded.paths, vectors.paths)
            self.assertEqual(loaded.metadata, vectors.metadata)
            self.assertEqual(loaded.class_names, vectors.class_names)
        with self.assertRaises(ValueError):
            FeatureVectors(vectors.features, np.array([0, 0, 1, 2]), vectors.paths, vectors.class_names, {})

    @patch("forensics_app.tools.set8.features.choose_model", return_value="")
    def test_cancel_returns_none_without_an_open_image(self, _choose) -> None:
        self.assertFalse(CNNFeatureExtractionTool.requires_image)
        self.assertIsNone(CNNFeatureExtractionTool().run(None, ImageDocument()))

    def test_tool_saves_features_with_model_provenance(self) -> None:
        vectors = FeatureVectors(np.ones((4, 3), dtype=np.float32), np.array([0, 0, 1, 1]),
                                 np.array(["a", "b", "c", "d"]), ("cat", "dog"), {"feature_layer": "features"})
        with TemporaryDirectory() as name:
            path = Path(name) / "features.npz"
            with patch("forensics_app.tools.set8.features.choose_model", return_value="model/passport.json"), patch(
                "forensics_app.tools.set8.features.filedialog.askdirectory", return_value="test"
            ), patch("forensics_app.tools.set8.features.simpledialog.askinteger", return_value=2), patch(
                "forensics_app.tools.set8.features.filedialog.asksaveasfilename", return_value=str(path)
            ), patch("forensics_app.tools.set8.features.load_cnn", return_value=(MagicMock(), {})), patch(
                "forensics_app.tools.set8.features.extract_features", return_value=vectors
            ) as extract:
                result = CNNFeatureExtractionTool().run(None, ImageDocument())
            self.assertEqual(extract.call_args.args[0], Path("test"))
            self.assertEqual(extract.call_args.args[3], 2)
            self.assertEqual(result.details["Test images"], 4)
            saved = load_features(path)
            self.assertEqual(saved.metadata["model_passport"], str(Path("model/passport.json").resolve()))

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
    def test_extraction_uses_test_images_in_batches_and_passport_label_order(self) -> None:
        model = build_cnn(2)
        passport = {"class_names": ["dog", "cat"], "feature_layer": "features"}
        with TemporaryDirectory() as name:
            root = Path(name)
            for label, color in (("cat", "red"), ("dog", "blue")):
                (root / label).mkdir()
                for index in range(3):
                    Image.new("RGB", (40, 24), color).save(root / label / f"{index}.png")
            vectors = extract_cnn_features(root, model, passport, batch_size=2)
            self.assertEqual(vectors.features.shape, (6, 100))
            self.assertEqual(vectors.features.dtype, np.float32)
            np.testing.assert_array_equal(vectors.labels, [0, 0, 0, 1, 1, 1])
            self.assertTrue(all(Path(path).parent.name == "dog" for path in vectors.paths[:3]))
            expected = get_keras().Model(inputs=model.inputs[0], outputs=model.get_layer("features").output)
            np.testing.assert_allclose(vectors.features[:1], expected(prepare_image(Image.new("RGB", (40, 24), "blue"))[None, ...], training=False).numpy(), atol=1e-6)
            (root / "unknown").mkdir()
            with self.assertRaisesRegex(ValueError, "Unknown test class"):
                extract_cnn_features(root, model, passport)


class CNNPredictionTests(unittest.TestCase):
    def test_passport_details_include_nested_preprocessing_and_history(self) -> None:
        details = passport_details({"class_names": ["cat", "dog"], "preprocessing": {"color_mode": "RGB"}, "training_history": {"loss": [0.5]}})
        self.assertEqual(details["Passport.preprocessing.color_mode"], "RGB")
        self.assertEqual(json.loads(details["Passport.class_names"]), ["cat", "dog"])
        self.assertEqual(json.loads(details["Passport.training_history.loss"]), [0.5])

    @patch("forensics_app.tools.set8.prediction.choose_model", return_value="")
    def test_cancel_returns_none(self, _choose) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (32, 32))
        self.assertIsNone(CNNPredictionTool().run(None, document))

    def test_predict_uses_rgb_normalization_and_passport_class_order(self) -> None:
        model = MagicMock()
        model.return_value.numpy.return_value = np.array([[0.1, 0.9]])
        image = Image.new("L", (64, 48), 255)
        label, scores = predict_cnn_image(image, model, {"class_names": ["dog", "cat"]})
        self.assertEqual(label, "cat")
        self.assertEqual(scores["cat"], 0.9)
        pixels = model.call_args.args[0]
        self.assertEqual(pixels.shape, (1, 32, 32, 3))
        self.assertEqual(pixels.dtype, np.float32)
        np.testing.assert_array_equal(pixels, 1)
        self.assertFalse(model.call_args.kwargs["training"])

    def test_tool_displays_passport_and_preserves_opened_image(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (40, 30), "blue")
        original = document.current.copy()
        model = MagicMock()
        model.return_value.numpy.return_value = np.array([[0.2, 0.8]])
        passport = {"class_names": ["cat", "dog"], "dataset": "training folder", "model_selection": {"best_epoch": 3}}
        with patch("forensics_app.tools.set8.prediction.choose_model", return_value="model/passport.json"), patch(
            "forensics_app.tools.set8.prediction.load_cnn", return_value=(model, passport)
        ):
            result = CNNPredictionTool().run(None, document)
        self.assertEqual(result.details["Prediction"], "dog")
        self.assertEqual(result.details["Passport.dataset"], "training folder")
        self.assertEqual(result.details["Passport.model_selection.best_epoch"], 3)
        self.assertEqual(result.image.mode, "RGB")
        np.testing.assert_array_equal(np.array(document.current), np.array(original))

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
    def test_saved_bundle_reload_preserves_predictions(self) -> None:
        model = build_cnn(2)
        passport = {
            "format": "classification_cnn_v1", "architecture": "architecture.json", "weights": "model.weights.h5",
            "class_names": ["cat", "dog"], "feature_layer": "features",
            "preprocessing": {"input_shape": [32, 32, 3], "color_mode": "RGB", "dtype": "float32", "normalization": "Divide pixel values by 255.0"},
        }
        image = Image.new("RGB", (48, 32), "red")
        with TemporaryDirectory() as name:
            folder = Path(name)
            save_cnn(folder, model, passport)
            loaded, saved_passport = load_cnn(folder / "passport.json")
            self.assertEqual(saved_passport, passport)
            _, expected = predict_cnn_image(image, model, passport)
            _, actual = predict_cnn_image(image, loaded, passport)
            np.testing.assert_allclose(list(actual.values()), list(expected.values()))
            passport["class_names"] = ["cat", "cat"]
            (folder / "passport.json").write_text(json.dumps(passport))
            with self.assertRaisesRegex(ValueError, "repeat"):
                load_cnn(folder / "passport.json")


class CNNTrainingTests(unittest.TestCase):
    def test_sample_limit_is_reproducible_and_keeps_smaller_classes(self) -> None:
        labels = np.array([0] * 12 + [1] * 3)
        settings = CNNSettings(samples_per_class=4)
        indices = select_sample_indices(labels, settings)
        np.testing.assert_array_equal(indices, select_sample_indices(labels, settings))
        np.testing.assert_array_equal(np.bincount(labels[indices]), [4, 3])
        np.testing.assert_array_equal(select_sample_indices(labels, CNNSettings()), np.arange(15))
        self.assertEqual(len(np.unique(indices)), len(indices))
        for limit in (-1, 1, 2.5):
            with self.assertRaises(ValueError):
                CNNSettings(samples_per_class=limit)

    def test_folder_sample_limit_is_applied_before_opening_images(self) -> None:
        with TemporaryDirectory() as name:
            root = Path(name)
            for label in ("cat", "dog"):
                (root / label).mkdir()
                for index in range(10):
                    Image.new("RGB", (32, 32), (index, 0, 0)).save(root / label / f"{index}.png")
            settings = CNNSettings(samples_per_class=2)
            with patch("forensics_app.tools.set8.classification.Image.open", wraps=Image.open) as opened:
                images, labels, names = load_training_images(root, settings)
                self.assertEqual(opened.call_count, 4)
            repeated, repeated_labels, _ = load_training_images(root, settings)
            self.assertEqual(names, ("cat", "dog"))
            np.testing.assert_array_equal(labels, [0, 0, 1, 1])
            np.testing.assert_array_equal(images, repeated)
            np.testing.assert_array_equal(labels, repeated_labels)

    @patch("forensics_app.tools.set8.classification.get_keras")
    @patch("forensics_app.tools.set8.classification.build_cnn")
    def test_training_sample_limit_precedes_validation_split_and_is_saved(self, build, _keras) -> None:
        build.return_value.fit.return_value.history = {"loss": [0.5], "accuracy": [0.7], "val_loss": [0.6], "val_accuracy": [0.5]}
        images = np.zeros((20, 32, 32, 3), dtype=np.float32)
        labels = np.array([0] * 10 + [1] * 10)
        _, passport = train_cnn(images, labels, ("cat", "dog"), "test", CNNSettings(epochs=1, samples_per_class=4, validation_fraction=0.25))
        self.assertEqual(passport["sample_size"], 8)
        self.assertEqual(passport["samples_per_class"], {"cat": 4, "dog": 4})
        self.assertEqual(passport["training_samples"], 6)
        self.assertEqual(passport["validation_samples"], 2)
        self.assertEqual(passport["settings"]["samples_per_class"], 4)

    @patch("builtins.print")
    def test_progress_logs_are_timestamped_and_flushed(self, output) -> None:
        log_progress("Loading weights...")
        self.assertIn("[Set 8] Loading weights...", output.call_args.args[0])
        self.assertTrue(output.call_args.kwargs["flush"])

    def test_registered_without_requiring_an_image(self) -> None:
        tools = dict(build_tool_registry().categories())["Set 8"]
        self.assertTrue(any(isinstance(tool, CNNTrainingTool) for tool in tools))
        self.assertFalse(CNNTrainingTool.requires_image)

    def test_settings_reject_invalid_values(self) -> None:
        for values in ({"epochs": 0}, {"batch_size": 0}, {"learning_rate": float("nan")},
                       {"validation_fraction": 1}, {"seed": -1}, {"selection_metric": "loss"},
                       {"architecture": "unknown"}, {"training_mode": "unknown"}, {"training_mode": "transfer"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CNNSettings(**values)

    def test_preprocessing_resizes_converts_rgb_and_normalizes(self) -> None:
        pixels = prepare_image(Image.new("L", (64, 48), 255))
        self.assertEqual(pixels.shape, (32, 32, 3))
        self.assertEqual(pixels.dtype, np.float32)
        np.testing.assert_array_equal(pixels, np.ones((32, 32, 3)))

    def test_folder_labels_follow_sorted_class_names(self) -> None:
        with TemporaryDirectory() as name:
            root = Path(name)
            for label in ("dog", "cat"):
                (root / label).mkdir()
                for index in range(2):
                    Image.new("RGB", (40, 30)).save(root / label / f"{index}.png")
            images, labels, names = load_training_images(root)
            self.assertEqual(names, ("cat", "dog"))
            self.assertEqual(images.shape, (4, 32, 32, 3))
            np.testing.assert_array_equal(labels, [0, 0, 1, 1])
            (root / "empty").mkdir()
            with self.assertRaisesRegex(ValueError, "At least two images"):
                load_training_images(root)

    @patch("forensics_app.tools.set8.classification.get_keras")
    @patch("forensics_app.tools.set8.classification.build_cnn")
    def test_training_uses_stratified_validation_and_saves_passport(self, build, keras) -> None:
        model = build.return_value
        model.fit.return_value.history = {
            "loss": [np.float32(0.7)], "accuracy": [0.5], "val_loss": [0.8], "val_accuracy": [0.4],
        }
        model.to_json.return_value = '{"class_name": "Sequential"}'
        images = np.zeros((20, 32, 32, 3), dtype=np.float32)
        labels = np.array([0] * 10 + [1] * 10)
        settings = CNNSettings(epochs=1, validation_fraction=0.2)
        trained, passport = train_cnn(images, labels, ("cat", "dog"), "test dataset", settings)
        self.assertIs(trained, model)
        build.assert_called_once_with(2, settings)
        training_labels = model.fit.call_args.args[1]
        validation_labels = model.fit.call_args.kwargs["validation_data"][1]
        np.testing.assert_array_equal(training_labels.sum(axis=0), [8, 8])
        np.testing.assert_array_equal(validation_labels.sum(axis=0), [2, 2])
        self.assertEqual(passport["sample_size"], 20)
        self.assertEqual(passport["training_samples"], 16)
        self.assertEqual(passport["validation_samples"], 4)
        self.assertEqual(passport["class_names"], ["cat", "dog"])
        model.load_weights.assert_called_once()
        checkpoint_options = keras.return_value.callbacks.ModelCheckpoint.call_args.kwargs
        self.assertEqual(checkpoint_options["monitor"], "val_loss")
        self.assertTrue(checkpoint_options["save_best_only"])
        self.assertEqual(passport["model_selection"]["best_epoch"], 1)
        self.assertEqual(passport["samples_per_class"], {"cat": 10, "dog": 10})
        with TemporaryDirectory() as name:
            root = Path(name)
            save_cnn(root, trained, passport)
            model.save_weights.assert_called_once_with(root / "model.weights.h5")
            saved = json.loads((root / "passport.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, passport)
            self.assertEqual(saved["preprocessing"]["input_shape"], [32, 32, 3])
            self.assertEqual(saved["settings"]["epochs"], 1)
            self.assertEqual(saved["training_history"]["accuracy"], [0.5])
            self.assertEqual(json.loads((root / "architecture.json").read_text())["class_name"], "Sequential")
        self.assertGreater(render_training_history(passport["training_history"]).width, 0)

    @patch("forensics_app.tools.set8.classification.get_keras")
    @patch("forensics_app.tools.set8.classification.build_cnn")
    def test_selection_uses_validation_metric_and_restores_checkpoint(self, build, keras) -> None:
        model = build.return_value
        model.fit.return_value.history = {
            "loss": [0.9, 0.5, 0.2], "accuracy": [0.4, 0.6, 0.9],
            "val_loss": [0.7, 0.4, 0.6], "val_accuracy": [0.8, 0.7, 0.6],
        }
        images = np.zeros((20, 32, 32, 3), dtype=np.float32)
        labels = np.array([0] * 10 + [1] * 10)
        for metric, epoch, mode in (("val_loss", 2, "min"), ("val_accuracy", 1, "max")):
            with self.subTest(metric=metric):
                _, passport = train_cnn(images, labels, ("cat", "dog"), "test", CNNSettings(epochs=3, selection_metric=metric))
                self.assertEqual(passport["model_selection"]["best_epoch"], epoch)
                self.assertEqual(passport["model_selection"]["mode"], mode)
                self.assertEqual(keras.return_value.callbacks.ModelCheckpoint.call_args.kwargs["monitor"], metric)
        self.assertEqual(model.load_weights.call_count, 2)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
    def test_saved_weights_are_the_best_epoch_not_the_last(self) -> None:
        keras = get_keras()
        images = np.zeros((20, 32, 32, 3), dtype=np.float32)
        labels = np.array([0] * 10 + [1] * 10)
        metrics = {
            "loss": [0.9, 0.5, 0.2], "accuracy": [0.4, 0.6, 0.9],
            "val_loss": [0.7, 0.4, 0.6], "val_accuracy": [0.8, 0.8, 0.6],
        }
        for metric, best_epoch in (("val_loss", 2), ("val_accuracy", 1)):
            with self.subTest(metric=metric):
                model = keras.Sequential([
                    keras.layers.Input(shape=(32, 32, 3)), keras.layers.Flatten(), keras.layers.Dense(2),
                ])

                def fit_epochs(*_args, **kwargs):
                    checkpoint = kwargs["callbacks"][0]
                    checkpoint.set_model(model)
                    for epoch in range(3):
                        model.set_weights([np.full_like(weight, epoch + 1) for weight in model.get_weights()])
                        checkpoint.on_epoch_end(epoch, {key: values[epoch] for key, values in metrics.items()})
                    return MagicMock(history=metrics)

                with patch("forensics_app.tools.set8.classification.build_cnn", return_value=model), patch.object(model, "fit", side_effect=fit_epochs):
                    trained, passport = train_cnn(images, labels, ("cat", "dog"), "test", CNNSettings(epochs=3, selection_metric=metric))
                self.assertEqual(passport["model_selection"]["best_epoch"], best_epoch)
                with TemporaryDirectory() as name:
                    root = Path(name)
                    save_cnn(root, trained, passport)
                    restored = keras.models.model_from_json((root / "architecture.json").read_text(encoding="utf-8"))
                    restored.load_weights(root / "model.weights.h5")
                    for weights in restored.get_weights():
                        np.testing.assert_array_equal(weights, np.full_like(weights, best_epoch))

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
    def test_small_and_vgg_architectures_match_the_notebook(self) -> None:
        keras = get_keras()
        for architecture, convolutions, units in (("small_cnn", 1, 100), ("vgg_cnn", 2, 128)):
            with self.subTest(architecture=architecture):
                model = build_cnn(3, CNNSettings(architecture=architecture))
                self.assertEqual(model.input_shape, (None, 32, 32, 3))
                self.assertEqual(model.output_shape, (None, 3))
                self.assertEqual(sum(isinstance(layer, keras.layers.Conv2D) for layer in model.layers), convolutions)
                self.assertEqual(model.get_layer("features").units, units)
                self.assertIsInstance(model.optimizer, keras.optimizers.SGD)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
    def test_model_details_include_layer_settings_shapes_and_parameter_counts(self) -> None:
        model = build_cnn(10)
        details = describe_model(model)
        self.assertEqual(details["Model input shape"], "(None, 32, 32, 3)")
        self.assertEqual(details["Model output shape"], "(None, 10)")
        self.assertEqual(details["Total parameters"], model.count_params())
        self.assertEqual(details["Trainable parameters"], model.count_params())
        self.assertEqual(details["Non-trainable parameters"], 0)
        layers = [value for key, value in details.items() if key.startswith("Layer ")]
        self.assertEqual(len(layers), 5)
        self.assertIn("Conv2D; output=(None, 30, 30, 32)", layers[0])
        self.assertIn('"kernel_size": [3, 3]', layers[0])
        self.assertIn('"activation": "relu"', layers[0])
        self.assertIn('"units": 100', layers[3])
        self.assertIn('"activation": "softmax"', layers[4])
        with TemporaryDirectory() as name:
            root = Path(name)
            save_cnn(root, model, {})
            summary = (root / "model_summary.txt").read_text(encoding="utf-8")
            self.assertIn("small_cnn", summary)
            self.assertIn("Conv2D", summary)
            self.assertIn("Total params", summary)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
    def test_inception_modes_resize_normalize_and_freeze_only_transfer_base(self) -> None:
        keras = get_keras()
        for mode, weights in (("scratch", None), ("transfer", "imagenet")):
            with self.subTest(mode=mode):
                inputs = keras.layers.Input(shape=(128, 128, 3))
                features = keras.layers.Conv2D(2, 1)(inputs)
                features = keras.layers.BatchNormalization()(features)
                base = keras.Model(inputs, features, name="inception_v3")
                with patch.object(keras.applications, "InceptionV3", return_value=base) as factory:
                    model = build_cnn(3, CNNSettings(architecture="inception_v3", training_mode=mode, learning_rate=0.001))
                factory.assert_called_once_with(include_top=False, weights=weights, input_shape=(128, 128, 3))
                self.assertEqual(base.trainable, mode == "scratch")
                self.assertEqual(model.input_shape, (None, 32, 32, 3))
                self.assertEqual(model.output_shape, (None, 3))
                self.assertEqual(model.get_layer("features").units, 1024)
                self.assertIsInstance(model.optimizer, keras.optimizers.Adam)
                preprocessing = keras.Model(model.inputs, model.get_layer("inception_preprocessing").output)
                np.testing.assert_allclose(preprocessing(np.zeros((1, 32, 32, 3))).numpy(), -1)
                np.testing.assert_allclose(preprocessing(np.ones((1, 32, 32, 3))).numpy(), 1)
                if mode == "transfer":
                    details = describe_model(model)
                    nested = [value for key, value in details.items() if key.startswith("Layer 4.")]
                    self.assertTrue(any("BatchNormalization" in value and "trainable=False" in value for value in nested))
                    self.assertGreater(details["Non-trainable parameters"], 0)
                    original = [weight.copy() for weight in base.get_weights()]
                    model.train_on_batch(np.ones((2, 32, 32, 3)), np.eye(3)[[0, 1]])
                    for before, after in zip(original, base.get_weights()):
                        np.testing.assert_array_equal(before, after)

    @patch("forensics_app.tools.set8.classification.choose_option", return_value=None)
    def test_cancel_returns_none(self, _choose) -> None:
        self.assertIsNone(CNNTrainingTool().run(None, ImageDocument()))

    @patch("forensics_app.tools.set8.classification.build_cnn")
    def test_invalid_labels_and_validation_split_fail_before_training(self, build) -> None:
        images = np.zeros((4, 32, 32, 3), dtype=np.float32)
        for labels, settings in ((np.array([0, 0, 1, 2]), CNNSettings()),
                                 (np.array([0, 0, 1, 1]), CNNSettings(validation_fraction=0.9))):
            with self.assertRaises(ValueError):
                train_cnn(images, labels, ("cat", "dog"), "test", settings)
        build.assert_not_called()

    @patch("forensics_app.tools.set8.classification.build_cnn")
    def test_empty_and_out_of_range_pixels_fail_before_training(self, build) -> None:
        for images, labels in ((np.empty((0, 32, 32, 3)), np.array([], dtype=int)),
                               (np.full((4, 32, 32, 3), 255), np.array([0, 0, 1, 1]))):
            with self.assertRaises(ValueError):
                train_cnn(images, labels, ("cat", "dog"), "test")
        build.assert_not_called()

    @patch("forensics_app.tools.set8.classification.describe_model", return_value={"Total parameters": 123})
    @patch("forensics_app.tools.set8.classification.get_keras")
    @patch("forensics_app.tools.set8.classification.choose_cnn_settings", return_value=CNNSettings(epochs=1))
    @patch("forensics_app.tools.set8.classification.choose_option", return_value="cifar10")
    @patch("forensics_app.tools.set8.classification.train_cnn")
    @patch("forensics_app.tools.set8.classification.save_cnn")
    def test_cifar_tool_uses_training_split_and_exports_results(self, save, train, _source, _settings, keras, describe) -> None:
        keras.return_value.datasets.cifar10.load_data.return_value = (
            (np.full((20, 32, 32, 3), 255, dtype=np.uint8), np.arange(20).reshape(-1, 1) % 10),
            (np.zeros((10, 32, 32, 3), dtype=np.uint8), np.arange(10).reshape(-1, 1)),
        )
        train.return_value = MagicMock(), {
            "training_samples": 10, "validation_samples": 10,
            "model_selection": {"best_epoch": 1},
            "training_history": {"loss": [1.0], "accuracy": [0.5], "val_loss": [1.2], "val_accuracy": [0.4]},
        }
        result = CNNTrainingTool().run(None, ImageDocument())
        self.assertEqual(train.call_args.args[0].shape, (20, 32, 32, 3))
        self.assertEqual(train.call_args.args[0].dtype, np.float32)
        self.assertEqual(float(train.call_args.args[0].min()), 1.0)
        self.assertEqual(train.call_args.args[1].shape, (20,))
        self.assertEqual(len(train.call_args.args[2]), 10)
        self.assertEqual(result.details["Training images"], 10)
        self.assertEqual(result.details["Validation accuracy"], 0.4)
        self.assertTrue(result.details["Weights"].endswith("model.weights.h5"))
        self.assertEqual(result.details["Total parameters"], 123)
        self.assertTrue(result.details["Model summary"].endswith("model_summary.txt"))
        describe.assert_called_once_with(train.return_value[0])
        save.assert_called_once()


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

    def test_feature_assignments_match_histogram_counts_and_descriptor_order(self) -> None:
        centers = np.array([[0, 0], [10, 10], [20, 20]], dtype=np.float32)
        regions = [np.array([[19, 19], [1, 1]]), np.empty((0, 2)), np.array([[9, 9], [18, 18]])]
        histogram, assignments = assign_features_to_words(regions, centers)
        np.testing.assert_array_equal(assignments, [2, 0, 1, 2])
        np.testing.assert_array_equal(histogram, [1, 0, 1, 0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(np.bincount(assignments, minlength=3), histogram.reshape(-1, 3).sum(axis=0))

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

    @patch("forensics_app.tools.set7.histogram.figure_to_image")
    def test_features_connect_to_assigned_visual_word_bars(self, render) -> None:
        positions = np.array([[4, 8], [12, 16], [20, 24]])
        assignments = np.array([2, 0, 2])
        render_histogram(
            Image.new("RGB", (32, 32)), np.array([1, 0, 2]),
            CodebookSettings(words=3, pyramid_levels=(1,)),
            positions=positions, assignments=assignments,
        )
        figure = render.call_args.args[0]
        image_axis, word_axis, _pyramid_axis = figure.axes
        self.assertEqual(len(image_axis.collections), 0)
        self.assertEqual(len(figure.artists), len(positions))
        for connection, position, word in zip(figure.artists, positions, assignments):
            np.testing.assert_array_equal(connection.xy1, position)
            np.testing.assert_allclose(connection.xy2, (word, word_axis.patches[word].get_height()))
            self.assertIs(connection.axesA, image_axis)
            self.assertIs(connection.axesB, word_axis)
            self.assertFalse(connection.get_in_layout())
            np.testing.assert_allclose(
                connection.get_edgecolor()[:3], word_axis.patches[word].get_facecolor()[:3],
            )

    @patch("forensics_app.tools.set7.histogram.choose_codebook", return_value="")
    def test_cancel_does_not_request_another_image(self, choose) -> None:
        document = MagicMock(current=Image.new("RGB", (64, 64)))
        self.assertIsNone(VisualWordHistogramTool().run(None, document))
        choose.assert_called_once_with(None)

    @patch("forensics_app.tools.set7.histogram.extract_pyramid_sift")
    @patch("forensics_app.tools.set7.histogram.load_codebook")
    @patch("forensics_app.tools.set7.histogram.choose_codebook", return_value="codebook.npz")
    def test_tool_uses_opened_image_and_returns_image_output(self, _choose, load, extract) -> None:
        model = CodebookModel(np.zeros((10, 128)), CodebookSettings())
        load.return_value = model
        extract.return_value = np.zeros((32, 32)), [cv2.KeyPoint(8, 12, 4)], [np.zeros((1, 128))] + [np.empty((0, 128))] * 20
        image = Image.new("RGB", (64, 64))
        document = MagicMock(current=image)
        result = VisualWordHistogramTool().run(None, document)
        extract.assert_called_once_with(image, model.settings)
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
                ("Set 8", ("cnn_training", "cnn_prediction", "cnn_feature_extraction", "cnn_feature_projection")),
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
