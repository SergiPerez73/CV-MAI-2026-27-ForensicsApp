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

## Course roadmap

The shell is intentionally neutral about the algorithms. Possible tool modules  
can follow the syllabus: filtering and edges, HOG, SIFT, bag of visual words,  
face detection/recognition, CNN classification and detection, transformers,  
semantic segmentation, and visualization/interpretability. Model weights and  
datasets should not be committed to the starter repository; document where they  
come from and keep paths configurable.
