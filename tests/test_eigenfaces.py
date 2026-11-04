import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.base import ToolResult
from forensics_app.tools.set6.eigenface_builder import EigenfaceBuilderTool, EigenfaceSettings, render_eigenfaces
from forensics_app.tools.set6.face_recognition import FaceRecognitionSettings, FaceRecognitionTool, render_face_matches
from forensics_app.tools.set6.eigenfaces import (
    fit_eigenfaces,
    inverse_distance_vote,
    load_eigenface_model,
    load_face_folder,
    nearest_faces,
    resolve_training_folder,
    save_eigenface_dataset,
)
from forensics_app.ui.main_window import MainWindow


class EigenfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.images = np.random.default_rng(42).integers(0, 256, (16, 5, 4)).astype(np.float32) / 255
        self.labels = np.array(["Person A"] * 8 + ["Person B"] * 8)

    def test_settings_only_include_components_and_face_size(self) -> None:
        settings = EigenfaceSettings()
        self.assertEqual(settings.components, 20)
        self.assertEqual(set(vars(settings)), {"components", "width", "height"})

    def test_pca_matches_pdf_projection_and_uses_training_mean(self) -> None:
        model = fit_eigenfaces(self.images[:12], 3)
        faces = self.images[:12].reshape(12, -1)
        self.assertEqual(model.components_.shape, (3, 20))
        self.assertFalse(model.whiten)
        np.testing.assert_allclose(model.mean_, faces.mean(axis=0))
        np.testing.assert_allclose(model.transform(faces), (faces - model.mean_) @ model.components_.T, atol=1e-6)

    def test_k_limits_and_identical_faces_are_rejected(self) -> None:
        for components in (0, 16, 100):
            with self.subTest(components=components), self.assertRaisesRegex(ValueError, "K"):
                fit_eigenfaces(self.images, components)
        with self.assertRaises(ValueError):
            fit_eigenfaces(np.zeros((3, 5, 4)), 1)
        model = fit_eigenfaces(self.images, 15)
        self.assertEqual(model.n_components_, 15)

    def test_folder_loading_resizes_grayscale_and_keeps_names(self) -> None:
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "Person A").mkdir()
            Image.new("RGB", (10, 10), "red").save(folder / "Person A" / "face.png")
            images, labels, paths = load_face_folder(folder, (4, 5))
            self.assertEqual(images.shape, (1, 5, 4))
            self.assertEqual(images.dtype, np.float32)
            self.assertEqual(labels.tolist(), ["Person A"])
            self.assertEqual(paths.tolist(), [str((folder / "Person A" / "face.png").resolve())])
            self.assertTrue(np.all((images >= 0) & (images <= 1)))
        with TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "No face images"):
            load_face_folder(Path(temporary), (4, 5))

    def test_tutorial_folders_exclude_non_faces_and_use_sample_names(self) -> None:
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ("faces", "non_faces"):
                (folder / name).mkdir()
                Image.new("L", (4, 5)).save(folder / name / f"{name}_01.png")
            images, labels, paths = load_face_folder(folder, (4, 5))
            self.assertEqual(len(images), 1)
            self.assertEqual(labels.tolist(), ["faces_01"])
            self.assertEqual(paths.tolist(), [str((folder / "faces" / "faces_01.png").resolve())])
            direct_images, direct_labels, direct_paths = load_face_folder(folder / "faces", (4, 5))
            np.testing.assert_array_equal(images, direct_images)
            np.testing.assert_array_equal(labels, direct_labels)
            np.testing.assert_array_equal(paths, direct_paths)

    def test_saved_model_and_manifest_preserve_training_order(self) -> None:
        model = fit_eigenfaces(self.images[:12], 3)
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            train_paths = np.array([str(folder / "existing_train" / f"{index}.png") for index in range(12)])
            test_paths = np.array([str(folder / "existing_test" / f"{index}.png") for index in range(4)])
            output = folder / "model"
            path = save_eigenface_dataset(
                output, self.images[:12], self.labels[:12], train_paths, self.labels[12:], test_paths, model,
            )
            with (output / "labels.json").open(encoding="utf-8") as stream:
                manifest = json.load(stream)
            self.assertEqual(len(manifest), 16)
            self.assertEqual(sum(entry["split"] == "test" for entry in manifest), 4)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["mean"], model.mean_)
                np.testing.assert_allclose(saved["eigenfaces"], model.components_)
                np.testing.assert_array_equal(saved["train_labels"], self.labels[:12])
                np.testing.assert_array_equal(saved["train_paths"], train_paths)
                np.testing.assert_array_equal(saved["test_paths"], test_paths)
                np.testing.assert_array_equal(saved["test_labels"], self.labels[12:])
                np.testing.assert_array_equal(saved["image_shape"], [5, 4])
                np.testing.assert_allclose(saved["coefficients"], model.transform(self.images[:12].reshape(12, -1)))
            self.assertEqual({entry.name for entry in output.iterdir()}, {"eigenfaces.npz", "labels.json"})
            self.assertEqual([entry["path"] for entry in manifest[:12]], train_paths.tolist())
            with self.assertRaisesRegex(ValueError, "empty"):
                save_eigenface_dataset(
                    output, self.images[:12], self.labels[:12], train_paths, self.labels[12:], test_paths, model,
                )

    def test_gallery_displays_all_components(self) -> None:
        model = fit_eigenfaces(self.images, 3)
        image = render_eigenfaces(model.components_, (5, 4))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (600, 250))
        self.assertGreater(np.array(image).std(), 0)
        with self.assertRaises(ValueError):
            render_eigenfaces(np.empty((0, 20)), (5, 4))
        image = render_eigenfaces(fit_eigenfaces(self.images, 13).components_, (5, 4))
        self.assertEqual(image.size, (800, 1000))

    def test_builder_is_registered_and_does_not_require_open_image(self) -> None:
        tools = dict(build_tool_registry().categories())["Set 6"]
        self.assertTrue(any(isinstance(tool, EigenfaceBuilderTool) for tool in tools))
        self.assertFalse(EigenfaceBuilderTool.requires_image)

    def test_main_window_accepts_gallery_without_open_image(self) -> None:
        window = MagicMock()
        window.document = ImageDocument()
        tool = EigenfaceBuilderTool()
        result = ToolResult(message="Built eigenfaces.", image=Image.new("RGB", (600, 250)))
        with patch.object(tool, "run", return_value=result):
            MainWindow.run_tool(window, tool)
        window._require_image.assert_not_called()
        self.assertEqual(window.document.current.size, (600, 250))
        window._refresh.assert_called_once_with()

    @patch("forensics_app.tools.set6.eigenface_builder.filedialog.askdirectory", return_value="")
    def test_cancel_dataset_returns_none(self, _directory) -> None:
        self.assertIsNone(EigenfaceBuilderTool().run(None, ImageDocument()))

    @patch("forensics_app.tools.set6.eigenface_builder.filedialog.askdirectory", side_effect=["train", ""])
    @patch("forensics_app.tools.set6.eigenface_builder.choose_eigenfaces")
    def test_cancel_test_folder_does_not_open_settings(self, settings, _directories) -> None:
        self.assertIsNone(EigenfaceBuilderTool().run(None, ImageDocument()))
        settings.assert_not_called()

    @patch("forensics_app.tools.set6.eigenface_builder.filedialog.askdirectory", side_effect=["train", "test"])
    @patch("forensics_app.tools.set6.eigenface_builder.choose_eigenfaces", return_value=None)
    @patch("forensics_app.tools.set6.eigenface_builder.load_face_folder")
    def test_cancel_settings_does_not_load_images(self, load, _settings, _directories) -> None:
        self.assertIsNone(EigenfaceBuilderTool().run(None, ImageDocument()))
        load.assert_not_called()

    @patch("forensics_app.tools.set6.eigenface_builder.choose_eigenfaces", return_value=EigenfaceSettings(3, 4, 5))
    @patch("forensics_app.tools.set6.eigenface_builder.filedialog.askdirectory")
    def test_builder_reuses_existing_split_and_displays_all_components(self, directories, _settings) -> None:
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for index, (image, name) in enumerate(zip(self.images, self.labels)):
                split = "train" if index < 12 else "test"
                person = folder / split / name
                person.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.uint8(image * 255)).save(person / f"{index}.png")
            directories.side_effect = [str(folder / "train"), str(folder / "test")]
            with patch("forensics_app.tools.set6.eigenface_builder.DEFAULT_DATA", folder), patch(
                "forensics_app.tools.set6.eigenface_builder.datetime"
            ) as clock:
                clock.now.return_value.strftime.return_value = "20261104_153000_123456"
                result = EigenfaceBuilderTool().run(None, ImageDocument())
            output = folder / "output_20261104_153000_123456"
            self.assertEqual(directories.call_count, 2)
            self.assertEqual(result.details["Components (K)"], 3)
            self.assertEqual(result.details["Eigenfaces displayed"], 3)
            self.assertEqual(result.details["Training images"], 12)
            self.assertEqual(result.details["Test images"], 4)
            self.assertTrue((output / "eigenfaces.npz").is_file())
            self.assertTrue((output / "labels.json").is_file())
            self.assertEqual(result.details["Model"], str(output / "eigenfaces.npz"))
            with np.load(output / "eigenfaces.npz") as saved:
                images, _, _ = load_face_folder(folder / "train", (4, 5))
                np.testing.assert_allclose(saved["mean"], images.reshape(12, -1).mean(axis=0))
                self.assertTrue(all(Path(path).is_relative_to(folder / "train") for path in saved["train_paths"]))
                self.assertTrue(all(Path(path).is_relative_to(folder / "test") for path in saved["test_paths"]))

    @patch("forensics_app.tools.set6.eigenface_builder.choose_eigenfaces", return_value=EigenfaceSettings(3, 4, 5))
    @patch("forensics_app.tools.set6.eigenface_builder.filedialog.askdirectory")
    def test_same_train_and_test_images_are_rejected(self, directories, _settings) -> None:
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            Image.new("L", (4, 5)).save(folder / "face.png")
            directories.side_effect = [str(folder), str(folder)]
            with self.assertRaisesRegex(ValueError, "same image paths"):
                EigenfaceBuilderTool().run(None, ImageDocument())


class FaceRecognitionTests(unittest.TestCase):
    def test_retrieval_count_defaults_to_three(self) -> None:
        self.assertEqual(FaceRecognitionSettings().count, 3)

    def test_inverse_distance_vote_normalizes_and_combines_repeated_labels(self) -> None:
        vote = inverse_distance_vote(np.array(["Alice", "Bob", "Bob"]), np.array([1.0, 1.5, 1.5]))
        np.testing.assert_allclose(vote.weights, [3 / 7, 2 / 7, 2 / 7])
        self.assertAlmostEqual(vote.weights.sum(), 1)
        self.assertAlmostEqual(sum(vote.scores.values()), 1)
        self.assertEqual(vote.label, "Bob")
        self.assertAlmostEqual(vote.scores["Bob"], 4 / 7)

    def test_exact_matches_share_all_vote_weight(self) -> None:
        vote = inverse_distance_vote(np.array(["Alice", "Bob", "Bob"]), np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(vote.weights, [0.5, 0.5, 0.0])
        self.assertEqual(vote.label, "Alice")
        vote = inverse_distance_vote(np.array(["Alice", "Bob"]), np.array([2.0, 0.0]))
        self.assertEqual(vote.label, "Bob")
        np.testing.assert_array_equal(vote.weights, [0.0, 1.0])

    def test_vote_is_scale_invariant_and_handles_tiny_distances(self) -> None:
        labels = np.array(["Alice", "Bob"])
        vote = inverse_distance_vote(labels, np.array([1e-310, 2e-310]))
        np.testing.assert_allclose(vote.weights, [2 / 3, 1 / 3])
        np.testing.assert_allclose(vote.weights, inverse_distance_vote(labels, np.array([1.0, 2.0])).weights)

    def test_vote_rejects_invalid_distances_or_label_counts(self) -> None:
        for labels, distances in (
            ([], []), (["Alice"], [1.0, 2.0]), (["Alice"], [-1]),
            (["Alice"], [np.nan]), (["Alice"], [np.inf]),
        ):
            with self.subTest(distances=distances), self.assertRaises(ValueError):
                inverse_distance_vote(np.array(labels), np.array(distances))

    def test_face_recognition_is_registered_in_set6(self) -> None:
        tools = dict(build_tool_registry().categories())["Set 6"]
        self.assertTrue(any(isinstance(tool, FaceRecognitionTool) for tool in tools))

    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.images = np.random.default_rng(42).integers(0, 256, (8, 5, 4), dtype=np.uint8)
        self.labels = np.array(["Alice"] * 4 + ["Bob"] * 4)
        paths = []
        for index, image in enumerate(self.images):
            path = self.folder / f"train_{index}.png"
            Image.fromarray(image).save(path)
            paths.append(str(path))
        self.paths = np.array(paths)
        self.pca = fit_eigenfaces(self.images.astype(np.float32) / 255, 3)
        self.model_path = save_eigenface_dataset(
            self.folder / "model", self.images.astype(np.float32) / 255, self.labels, self.paths,
            np.array(["Alice"]), np.array([str(self.folder / "test.png")]), self.pca,
        )
        self.model = load_eigenface_model(self.model_path)

    def test_projection_and_distances_match_sklearn(self) -> None:
        image = Image.fromarray(self.images[2])
        indices, distances = nearest_faces(image, self.model)
        coefficients = self.pca.transform(self.images[2].astype(np.float32).reshape(1, -1) / 255)[0]
        expected = np.linalg.norm(self.model.coefficients - coefficients, axis=1)
        np.testing.assert_array_equal(indices, np.argsort(expected)[:3])
        np.testing.assert_allclose(distances, expected[indices], atol=1e-6)
        self.assertEqual(indices[0], 2)
        self.assertLess(distances[0], 1e-6)

    def test_preprocessing_uses_saved_dimensions_and_grayscale(self) -> None:
        image = Image.new("RGB", (40, 50), "red")
        indices, distances = nearest_faces(image, self.model)
        resized = np.array(image.convert("L").resize((4, 5), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        coefficients = self.pca.transform(resized.reshape(1, -1))[0]
        expected = np.linalg.norm(self.model.coefficients - coefficients, axis=1)
        np.testing.assert_allclose(distances, expected[indices], atol=1e-6)

    def test_match_count_is_limited_to_available_training_images(self) -> None:
        indices, distances = nearest_faces(Image.fromarray(self.images[0]), self.model, count=20)
        self.assertEqual(len(indices), 8)
        self.assertTrue(np.all(np.diff(distances) >= 0))
        with self.assertRaises(ValueError):
            nearest_faces(Image.fromarray(self.images[0]), self.model, count=0)

    def test_model_round_trip_and_relative_paths(self) -> None:
        np.testing.assert_allclose(self.model.mean, self.pca.mean_)
        np.testing.assert_array_equal(self.model.labels, self.labels)
        self.assertEqual(self.model.image_shape, (5, 4))
        with np.load(self.model_path) as saved:
            arrays = {name: saved[name] for name in saved.files}
        arrays["train_paths"] = np.array([f"../train_{index}.png" for index in range(8)])
        relative_model = self.folder / "model" / "relative.npz"
        np.savez_compressed(relative_model, **arrays)
        model = load_eigenface_model(relative_model)
        self.assertEqual([path.resolve() for path in model.paths], [Path(path) for path in self.paths])

    def test_missing_fields_invalid_shapes_and_nonfinite_models_are_rejected(self) -> None:
        invalid_path = self.folder / "invalid.npz"
        np.savez_compressed(invalid_path, mean=self.pca.mean_)
        with self.assertRaisesRegex(ValueError, "missing"):
            load_eigenface_model(invalid_path)
        with np.load(self.model_path) as saved:
            arrays = {name: saved[name] for name in saved.files}
        for field, value in (
            ("image_shape", np.array([0, 4])),
            ("image_shape", np.array([5.0, 4.0])),
            ("coefficients", np.ones((8, 2))),
            ("mean", np.full(20, np.nan)),
            ("train_labels", np.array(["Alice"])),
        ):
            with self.subTest(field=field):
                np.savez_compressed(invalid_path, **{**arrays, field: value})
                with self.assertRaises(ValueError):
                    load_eigenface_model(invalid_path)

    def test_gallery_and_missing_training_image(self) -> None:
        image = Image.fromarray(self.images[0])
        indices, distances = nearest_faces(image, self.model)
        output = render_face_matches(image, self.model, indices, distances)
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, (1200, 350))
        self.assertGreater(np.array(output).std(), 0)
        self.model.paths[int(indices[0])].unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Training image not found"):
            render_face_matches(image, self.model, indices, distances)

    @patch("forensics_app.tools.set6.face_recognition.filedialog.askopenfilename", return_value="")
    def test_cancel_model_returns_none(self, _choose) -> None:
        document = ImageDocument()
        document.current = Image.fromarray(self.images[0])
        self.assertIsNone(FaceRecognitionTool().run(None, document))

    @patch("forensics_app.tools.set6.face_recognition.filedialog.askdirectory", return_value="")
    @patch("forensics_app.tools.set6.face_recognition.filedialog.askopenfilename")
    def test_cancel_training_folder_returns_none(self, choose, _folder) -> None:
        choose.return_value = str(self.model_path)
        document = ImageDocument()
        document.current = Image.fromarray(self.images[0])
        self.assertIsNone(FaceRecognitionTool().run(None, document))

    @patch("forensics_app.tools.set6.face_recognition.filedialog.askdirectory")
    @patch("forensics_app.tools.set6.face_recognition.filedialog.askopenfilename")
    @patch("forensics_app.tools.set6.face_recognition.choose_face_recognition", return_value=FaceRecognitionSettings(5))
    def test_tool_uses_current_image_and_only_selects_model_and_training_folder(self, _settings, choose, folder) -> None:
        document = ImageDocument()
        filename = self.folder / "test.png"
        Image.fromarray(self.images[7]).save(filename)
        document.load(filename)
        document.apply(Image.fromarray(self.images[2]))
        choose.return_value = str(self.model_path)
        folder.return_value = str(self.folder)
        result = FaceRecognitionTool().run(None, document)
        self.assertEqual(result.details["Nearest name"], "Alice")
        self.assertEqual(result.details["Match 1"], "Alice")
        self.assertLess(result.details["Distance 1"], 1e-6)
        self.assertEqual(result.details["Components (K)"], 3)
        self.assertIn("Match 3", result.details)
        self.assertIn("Match 5", result.details)
        self.assertEqual(result.details["Retrieved matches"], 5)
        self.assertEqual(result.details["Requested matches"], 5)
        self.assertEqual(result.details["Vote result"], "Alice")
        self.assertAlmostEqual(result.details["Weight sum"], 1)
        self.assertAlmostEqual(sum(result.details[f"Weight {rank}"] for rank in range(1, 6)), 1)
        self.assertEqual(result.image.size, (1200, 700))
        self.assertEqual(result.details["Training folder"], str(self.folder))
        self.assertEqual(result.image.mode, "RGB")
        self.assertTrue(FaceRecognitionTool.requires_image)
        self.assertEqual(choose.call_count, 1)
        self.assertEqual(folder.call_count, 1)
        np.testing.assert_array_equal(np.array(document.current), self.images[2])

    @patch("forensics_app.tools.set6.face_recognition.filedialog.askdirectory")
    @patch("forensics_app.tools.set6.face_recognition.filedialog.askopenfilename")
    @patch("forensics_app.tools.set6.face_recognition.choose_face_recognition", return_value=None)
    def test_cancel_recognition_settings_returns_none(self, _settings, choose, folder) -> None:
        choose.return_value = str(self.model_path)
        folder.return_value = str(self.folder)
        document = ImageDocument()
        document.current = Image.fromarray(self.images[0])
        self.assertIsNone(FaceRecognitionTool().run(None, document))

    def test_training_folder_can_relocate_images_without_reordering_coefficients(self) -> None:
        folder = self.folder / "moved" / "faces"
        for path, label in zip(self.model.paths, self.model.labels):
            target = folder / str(label) / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
        model = resolve_training_folder(self.model, folder.parent)
        self.assertTrue(all(path.is_relative_to(folder) for path in model.paths))
        self.assertTrue(all(path.parent.name == label for path, label in zip(model.paths, model.labels)))
        self.assertIs(model.coefficients, self.model.coefficients)
        indices, distances = nearest_faces(Image.fromarray(self.images[2]), model)
        self.assertEqual(indices[0], 2)
        self.assertLess(distances[0], 1e-6)
        self.assertEqual(render_face_matches(Image.fromarray(self.images[2]), model, indices, distances).mode, "RGB")

    def test_incomplete_or_ambiguous_training_folder_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cannot uniquely locate"):
            resolve_training_folder(self.model, self.folder / "missing")
        for subfolder in ("duplicate1", "duplicate2"):
            target = self.folder / subfolder
            target.mkdir()
            Image.fromarray(self.images[0]).save(target / "train_0.png")
        with self.assertRaisesRegex(ValueError, "Cannot uniquely locate"):
            resolve_training_folder(self.model, self.folder)