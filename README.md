# ForensicsApp course starter

ForensicsApp is the semester project shell for the MSc Computer Vision course.  
Students add a small, demonstrable image-forensics feature as the course moves  
from classical image processing to deep learning and interpretability.

The starter already provides:

- a Tkinter desktop interface with tool, image, and results areas;
- opening and exporting common image formats;
- a fit-to-window image preview;
- undo, redo, and reset-to-original behavior;
- a tiny plug-in-style contract for adding course functionality;
- one analysis example and one image-processing example.

## Quick start

Python 3.10 or newer is recommended.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

On some Linux distributions, Tkinter is a separate system package. For example,  
Ubuntu/Debian users may first need `sudo apt install python3-tk`.

The complete student handout is [docs/STUDENT_GUIDE.md](docs/STUDENT_GUIDE.md).

## Useful commands

```
python -m forensics_app
python -m unittest discover -s tests -v
python -m compileall forensics_app
```

## Set 6: Face detection

Open an image and select **Set 6 > Face detection**. Choose **Haar / AdaBoost**
for OpenCV's pretrained frontal-face cascade, or **YuNet** for face boxes,
confidence scores, and five facial landmarks. Both methods have configurable
detection settings and preserve the original image dimensions.

The Haar cascade is included with `opencv-python`. YuNet defaults to the local
model at `sets/Se6_images/face_detection_yunet_2023mar.onnx`; use **Browse...** to
select another location. If needed, obtain the model from
[OpenCV Zoo](https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx).
The tool does not download weights automatically. This tool detects faces in
the current image; it does not identify people or capture webcam video.

## Set 6: Eigenface builder

Select **Set 6 > Eigenface builder**; no open image is needed. Select your
existing training folder and test folder separately. The selectors initially
open at `sets/Set6_images/images/train_data/faces` and
`sets/Set6_images/images/test_data/faces`. There is no LFW or download option, and the
builder never re-splits or copies the images.

When a selected folder contains `faces/` and `non_faces/`, only `faces/` is
loaded. Flat face folders use filename stems as sample identifiers; folders
organized by person use the person subfolder names as labels. The tutorial's
face/non-face dataset does not supply person identities, so its sample
identifiers cannot be used as person names for recognition. Local images must
already be cropped and aligned faces; the builder resizes them but does not
detect or align faces automatically.

Set **PCA components (K)**, defaulting to 20, and the face width and height,
defaulting to 37 x 50 grayscale pixels. All K eigenfaces are displayed; there
is no separate preview-count setting. K must be between 1 and the smaller of
the pixel count and training-image count minus one. Results include the
percentage of variance explained by the retained components.

Each build automatically creates a folder at
`sets/Set6_images/images/output_YYYYMMDD_HHMMSS_microseconds`; there is no
output-folder prompt. The builder writes only `labels.json`
with original image paths, names/sample identifiers, and train/test membership,
and `eigenfaces.npz` with the training mean, eigenfaces, training PCA
coefficients, train/test labels and absolute image paths, image shape, and
explained-variance ratios. Keep the original images at those paths for point 3.
Only training images are used to fit PCA; whitening is disabled to match the
PDF's projection formula. Images are represented as grayscale values in [0, 1].
This tool prepares the model but does not perform recognition itself.

## Set 6: Face recognition

Open a cropped test face image, then select **Set 6 > Face recognition**.
Choose an `eigenfaces.npz` saved by the builder and its training image folder.
The currently displayed image, including any applied edits, is the query;
there is no test-image picker. The model picker initially opens in the latest
timestamped output folder, and the training-folder picker opens in
`sets/Set6_images/images/train_data/faces`.

Set **Similar faces to retrieve**, defaulting to 3, to any positive whole
number. If fewer training samples exist, all available samples are retrieved.

The tool converts the test face to grayscale, resizes it to the saved model's
dimensions, scales pixels to [0, 1], subtracts the saved training mean, and
projects onto the saved eigenfaces. It compares these coefficients to the
training coefficients using Euclidean distance, without re-training PCA.

The gallery shows the test face and the requested closest training images,
with their names, distances, and normalized voting weights. Larger result
sets wrap into multiple rows. Each positive-distance weight is proportional
to `1 / distance`, with all image weights summing to 1. Weights are summed
per label, and the label with the highest total is the voting result.
Exact zero-distance matches share all the weight equally, with other matches
receiving zero. Equal label totals favor the label encountered first in the
nearest-first results. The gallery and results show the winner and its vote
share; results also show every per-label score and per-image weight.

Results also include the resolved training image paths. Names are read from
the model, so a separate label-file selection is unnecessary. Training images
are located within the selected folder by saved filenames, using person
subfolders to disambiguate duplicate filenames. Select the same dataset used
to build the model; missing or ambiguous files produce an error. The dataset
may be moved, provided its filenames and person folders are preserved.
The vote share is not a calibrated confidence or an unknown-person rejection
decision. No download is performed.

## Set 7: Codebook construction

Select **Set 7 > Codebook construction**; no open image is needed. Choose
`sets/Set7_images/train`, containing `butterfly/`, `dalmatian/`, and `kangaroo/`.
Only this training folder is read. The separate `test/` folder has the same
three class subfolders and is reserved for subsequent histogram visualization
and classification.

The dataset uses a reproducible, per-class 80/20 split (random seed 42):

| Class | Train | Test |
| --- | ---: | ---: |
| butterfly | 73 | 18 |
| dalmatian | 54 | 13 |
| kangaroo | 69 | 17 |

The settings dialog uses these parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `words` | 10 | Number of visual words, K |
| `pyramid_levels` | `(1, 4, 16)` | Number of crops at each level, giving 21 crops |
| `step` | 8 | Dense grid spacing in pixels at each octave's resolution |
| `sift_size` | 8 | SIFT spatial bin width in pixels at octave 0, scale 0 |
| `n_octaves` | 3 | Number of Gaussian octaves |
| `n_scales` | 3 | Scale layers per octave |
| `sigma_min` | 1.6 | Base Gaussian blur |
| `max_image_size` | 256 | Maximum resized image dimension |

The SIFT parameter names `n_octaves`, `n_scales`, and `sigma_min` match Set 5.
Unlike Set 5's detected keypoints, this tool uses an upright dense grid so
`step` controls sampling. Images keep their aspect ratio and are never upscaled.
The default K remains 10 rather than the PDF's `10 * number_of_classes` rule.

Processing is separated into these functions:

1. `prepare_image` converts to grayscale and resizes once.
2. `split_pyramid` makes independent crops: the whole image, four quadrants,
	and sixteen cells by default. Region counts must be perfect squares.
3. `extract_sift` computes local 128-value SIFT descriptors within each crop,
	using the configured grid, size, octaves, scales, and blur.
4. `extract_pyramid_sift` runs extraction on every crop, retaining region order.
5. `build_codebook` learns K centers from the training descriptors using MiniBatchKMeans.
6. `compute_histograms` assigns each crop's descriptors to the nearest centers
	and concatenates its K-bin word counts. Empty crops contribute K zeros.
7. `extract_phow` encodes a new image using the same settings and learned centers.

Thus `(1, 4, 16)` produces a `21 * K` image vector, or 210 values with K=10.
Each individual SIFT descriptor still has 128 values. Histograms saved by
codebook construction are raw counts, without normalization or level weighting.
Classification is trained separately.

Within each crop, bin width and Gaussian blur increase by
`2 ** (octave + scale / n_scales)`. The grid step doubles per octave.
OpenCV computes gradients in the explicitly selected Gaussian octave and layer.
Scales that cannot fit their spatial bin centers inside a crop contribute no
descriptors. OpenCV handles border samples within the wider interpolation support.

Each cluster center is a visual word. The gallery adjusts to K and shows
the training patch whose descriptor is closest to each center; the patches
are representatives, not images reconstructed from descriptor coordinates.
They show the nominal four-bin-wide region from the source pyramid crop,
not the blurred Gaussian image or the full interpolation support.

Each build saves `codebook.npz` in a new
`sets/Set7_images/output_YYYYMMDD_HHMMSS_microseconds` folder. It contains the
K-by-128 centroid matrix (`centers`), representative `patches`, `train_labels`,
`train_paths`, and `train_histograms` with one `sum(pyramid_levels) * K` vector
per training image. All extraction parameters from the table are saved, except
`words`, which is given by the number of centers. The `descriptor_method` value
`spatial_pyramid_dense_sift_v3` distinguishes this pipeline from older archives.
Rebuild old codebooks before using them with this extractor. Test images are
never used to learn the vocabulary or training histograms.
OpenCV and scikit-learn provide extraction and learning. Joblib stores classifiers.

## Set 7: Visual word histogram

Open a test image, then select **Set 7 > Visual word histogram** and choose a
saved visual-word model (`codebook.npz`). No image picker is shown. Extraction
uses the settings saved in that codebook; outdated models must be rebuilt.

The image output shows the opened image, a normalized K-bin histogram pooled
across regions, and the complete normalized spatial-pyramid vector. With the
default levels and K=10, the latter has 210 bins. Empty regions contribute zeros.
The result replaces the current image in the main view. Use Undo to restore it.

## Set 7: Predict class

Open an image and select **Set 7 > Predict class**. Choose either:

1. **Train and save classifier**: select a visual-word model, then a training
	folder containing `butterfly/`, `dalmatian/`, and `kangaroo/`, and a destination
	for the `.joblib` classifier. Each class needs at least two training images.
	Only the selected training folders are read; test images are not used in fitting.
2. **Load saved classifier**: select a previously saved `.joblib` file. No training
	folder or separate visual-word model is needed. Only load files you created or
	trust: joblib uses pickle and loading an untrusted file can execute Python code.

Training extracts pyramid histograms using the selected codebook, applies L1
normalization and an additive chi-squared feature map, then fits a class-balanced
linear SVM with probability calibration. The saved classifier includes the
codebook centers, extraction settings, normalization, feature map, and fitted SVM.
Reusing it does not depend on the original codebook file or training image paths.

Prediction shows the opened image and histograms with the highest-probability
class and its estimated probability. All three probabilities appear in the
tool details. The result replaces the current image in the main view, and Undo
restores the previous image. Images producing no descriptors are
rejected rather than classified from an empty feature vector.

## Course roadmap

The shell is intentionally neutral about the algorithms. Possible tool modules  
can follow the syllabus: filtering and edges, HOG, SIFT, bag of visual words,  
face detection/recognition, CNN classification and detection, transformers,  
semantic segmentation, and visualization/interpretability. Model weights and  
datasets should not be committed to the starter repository; document where they  
come from and keep paths configurable.
