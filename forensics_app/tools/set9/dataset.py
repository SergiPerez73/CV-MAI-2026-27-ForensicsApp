"""Convert the Set 9 blood-cell COCO export to YOLO detection labels."""

from __future__ import annotations

from collections import defaultdict
import json
from math import isfinite
from pathlib import Path
from shutil import copyfile


CLASS_NAMES = ("PC", "RBC", "WBC")
SPLITS = {"train": "train", "val": "valid", "test": "test"}
ANNOTATIONS = "_annotations.coco.json"


def coco_to_yolo(input_folder: Path, output_folder: Path) -> int:
    with (input_folder / ANNOTATIONS).open(encoding="utf-8-sig") as file:
        data = json.load(file)

    images = {image["id"]: image for image in data["images"]}
    if not images:
        raise ValueError(f"No annotated image records in {input_folder}.")
    categories = {entry["id"]: entry["name"] for entry in data["categories"]}
    labels: dict[int, list[str]] = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation.get("iscrowd"):
            continue
        image_id = annotation["image_id"]
        image = images[image_id]
        class_name = categories[annotation["category_id"]]
        if class_name not in CLASS_NAMES:
            raise ValueError(f"Unexpected annotated class {class_name!r} in {input_folder}.")
        width, height = image["width"], image["height"]
        left, top, box_width, box_height = annotation["bbox"]
        if width <= 0 or height <= 0 or not all(map(isfinite, annotation["bbox"])):
            raise ValueError(f"Invalid dimensions or box for {image['file_name']}.")
        right, bottom = min(left + box_width, width), min(top + box_height, height)
        left, top = max(left, 0), max(top, 0)
        if right <= left or bottom <= top:
            continue
        coords = ((left + right) / (2 * width), (top + bottom) / (2 * height),
                  (right - left) / width, (bottom - top) / height)
        labels[image_id].append(f"{CLASS_NAMES.index(class_name)} " + " ".join(f"{value:.10g}" for value in coords))

    files = set()
    for image in images.values():
        name = Path(image["file_name"])
        stem = name.stem.casefold()
        if name.name != image["file_name"] or not (input_folder / name).is_file() or stem in files:
            raise ValueError(f"Missing, invalid, or duplicate image: {input_folder / name}")
        files.add(stem)

    output_folder.mkdir(parents=True, exist_ok=False)
    image_dir, label_dir = output_folder / "images", output_folder / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    for image_id, image in images.items():
        name = Path(image["file_name"])
        copyfile(input_folder / name, image_dir / name)
        (label_dir / name.with_suffix(".txt")).write_text("\n".join(labels[image_id]), encoding="utf-8")
    return len(images)


def prepare_dataset(source: Path, destination: Path) -> Path:
    from yaml import safe_dump

    destination = destination.resolve()
    for split in SPLITS.values():
        if not (source / split / ANNOTATIONS).is_file():
            raise FileNotFoundError(f"Expected {source / split / ANNOTATIONS}.")

    destination.mkdir(parents=True, exist_ok=False)
    for split in SPLITS.values():
        coco_to_yolo(source / split, destination / split)

    config = {"path": destination.as_posix(), **{key: f"{split}/images" for key, split in SPLITS.items()},
              "nc": len(CLASS_NAMES), "names": list(CLASS_NAMES)}
    config_path = destination / "data.yaml"
    config_path.write_text(safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path