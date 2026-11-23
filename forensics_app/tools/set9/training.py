"""Train a YOLO11n detector from scratch on the Set 9 blood-cell dataset."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, SimpleQueue
from tempfile import mkdtemp
import tkinter as tk
from tkinter import filedialog, ttk

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..utils import add_dialog_buttons, add_numeric_field, create_dialog, wait_for_dialog
from .dataset import CLASS_NAMES, prepare_dataset


MODEL_CONFIG = "yolo11n.yaml"
ASSETS = Path(__file__).resolve().parents[3] / "sets" / "set9_images"


def train_detector(source: Path, output: Path, *, epochs: int, batch: int,
                   progress: Callable[[str], None]) -> ToolResult:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError('Install YOLO support with: python -m pip install -e ".[yolo]"') from error

    run_dir = Path(mkdtemp(prefix="blood_cells_", dir=output.resolve()))
    progress("Converting COCO annotations and copying images...")
    config = prepare_dataset(source, run_dir / "dataset")
    device = "0" if torch.cuda.is_available() else "cpu"
    progress(f"Starting YOLO11n training on {'CUDA' if device == '0' else 'CPU'}...")
    model = YOLO(MODEL_CONFIG)

    def report_epoch(trainer) -> None:
        progress(f"Completed epoch {trainer.epoch + 1} of {epochs}.")

    model.add_callback("on_train_epoch_end", report_epoch)
    model.train(data=str(config), project=str(run_dir), name="training", epochs=epochs, batch=batch,
                device=device, workers=0, pretrained=False, plots=True)
    checkpoint = run_dir / "training" / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"Training finished without the expected checkpoint: {checkpoint}")

    return ToolResult(message="Blood-cell detector saved.", details={
        "Checkpoint": str(checkpoint),
        "Dataset configuration": str(config),
        "Training logs and plots": str(checkpoint.parent.parent),
        "Classes": ", ".join(CLASS_NAMES),
        "Epochs requested": epochs,
        "Batch size": batch,
        "Device": "CUDA" if device == "0" else "CPU",
    })


def choose_training_settings(parent: tk.Misc) -> tuple[int, int] | None:
    dialog = create_dialog(parent, title="Blood-cell training settings")
    epochs = tk.StringVar(master=dialog, value="50")
    batch = tk.StringVar(master=dialog, value="16")
    error = tk.StringVar(master=dialog)
    selected = None

    def apply_settings() -> None:
        nonlocal selected
        try:
            epoch_count, batch_size = int(epochs.get()), int(batch.get())
            if min(epoch_count, batch_size) < 1:
                raise ValueError
        except ValueError:
            error.set("Enter positive whole numbers for epochs and batch size.")
            return
        selected = (epoch_count, batch_size)
        dialog.destroy()

    add_numeric_field(dialog, "Epochs:", epochs).configure(from_=1, to=1000)
    add_numeric_field(dialog, "Batch size:", batch).configure(from_=1, to=256)
    ttk.Label(dialog, textvariable=error).pack(padx=16, pady=(8, 0))
    add_dialog_buttons(dialog, apply_settings)
    wait_for_dialog(parent, dialog)
    return selected


class BloodCellTrainingTool(ForensicsTool):
    tool_id = "yolo_blood_cell_training"
    title = "Train blood-cell detector (YOLO)"
    category = "Set 9"
    description = "Convert a COCO dataset and train a YOLO11n blood-cell detector from scratch."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = filedialog.askdirectory(parent=parent, title="Choose the blood-cell COCO dataset",
                                         initialdir=ASSETS, mustexist=True)
        if not source:
            return None
        output = filedialog.askdirectory(parent=parent, title="Choose where to save training runs",
                                         initialdir=ASSETS, mustexist=True)
        if not output:
            return None
        settings = choose_training_settings(parent)
        if settings is None:
            return None

        epochs, batch = settings
        dialog = create_dialog(parent, title=self.title)
        status = tk.StringVar(master=dialog, value="Preparing training...")
        ttk.Label(dialog, textvariable=status, width=58).pack(padx=16, pady=16)
        dialog.protocol("WM_DELETE_WINDOW", lambda: None)
        dialog.grab_set()
        updates: SimpleQueue[str] = SimpleQueue()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(train_detector, Path(source), Path(output), epochs=epochs, batch=batch,
                                     progress=updates.put)

            def poll() -> None:
                try:
                    while True:
                        status.set(updates.get_nowait())
                except Empty:
                    pass
                if future.done():
                    dialog.destroy()
                else:
                    dialog.after(100, poll)

            dialog.after(100, poll)
            parent.wait_window(dialog)
            return future.result()