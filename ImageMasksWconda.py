"""
Cellpose GPU Segmentation - folder batch runner with color overlays.

Tuning notes:
  - CELL_DIAMETER: measure a typical cell in pixels. 0 = auto (unreliable
    on sparse / faint fluorescence).
  - CELLPROB_THRESHOLD: lower (-1 .. -3) to recover faint cells.
  - FLOW_THRESHOLD: raise (0.6 .. 0.9) to keep irregular / branched shapes.
  - CHAN / CHAN2: 0=grayscale, 1=red, 2=green, 3=blue. CHAN2=0 means no
    separate nuclear channel.
"""

import sys
import subprocess
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

import torch
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# ---- Segmentation parameters -------------------------------------------------
MODEL = "cyto3"            # "cyto3" (recommended) / "cyto2" / "cyto" / "nuclei"
CELL_DIAMETER = 60.0       # pixels; set 0.0 for auto-estimate
CHAN = 0                   # signal channel
CHAN2 = 0                  # nuclear channel, 0 = none
FLOW_THRESHOLD = 0.7
CELLPROB_THRESHOLD = -2.0
NORM_LOW, NORM_HIGH = 1.0, 99.0
# -----------------------------------------------------------------------------


def find_mask(image_path: Path, output_dir: Path):
    """Return the mask file matching this image, or None."""
    candidates = sorted(output_dir.glob(f"{image_path.stem}*_cp_masks*"))
    candidates = [c for c in candidates if c.suffix.lower() in IMAGE_EXTS]
    return candidates[0] if candidates else None


def create_color_overlay(image_path, output_dir):
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    mask_path = find_mask(image_path, output_dir)
    if mask_path is None:
        print(f"No mask found for {image_path.name}")
        return None

    img = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))

    n_cells = int(mask.max())
    # Hide background (label 0) so only cells get tinted.
    mask = np.ma.masked_where(mask == 0, mask)

    h, w = img.shape[:2]
    dpi = 200
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.imshow(mask, cmap="nipy_spectral", alpha=0.5, interpolation="nearest")
    ax.axis("off")

    out_path = output_dir / f"{image_path.stem}_overlay.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"{image_path.name}: {n_cells} cells -> {out_path.name}")
    return out_path


def build_command(folder_path: Path, output_dir: Path, use_gpu: bool):
    cmd = [
        sys.executable, "-m", "cellpose",
        "--dir", str(folder_path),
        "--pretrained_model", MODEL,
        "--diameter", str(CELL_DIAMETER),
        "--chan", str(CHAN),
        "--chan2", str(CHAN2),
        "--flow_threshold", str(FLOW_THRESHOLD),
        "--cellprob_threshold", str(CELLPROB_THRESHOLD),
        "--norm_percentile", str(NORM_LOW), str(NORM_HIGH),
        "--save_png",
        "--save_tif",
        "--verbose",
        "--savedir", str(output_dir),
    ]
    if use_gpu:
        cmd.append("--use_gpu")
    return cmd


def run_cellpose_on_folder(folder_path):
    folder_path = Path(folder_path)
    output_dir = folder_path / "output"
    output_dir.mkdir(exist_ok=True)

    images = [p for p in sorted(folder_path.iterdir())
              if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        messagebox.showwarning("No images", f"No images found in:\n{folder_path}")
        return

    cuda_available = torch.cuda.is_available()
    cmd = build_command(folder_path, output_dir, cuda_available)

    try:
        device_text = torch.cuda.get_device_name(0) if cuda_available else "CPU"
        messagebox.showinfo(
            "Cellpose",
            f"Starting segmentation on {len(images)} image(s)\n\n"
            f"Device: {device_text}\n"
            f"Model: {MODEL}\n"
            f"Diameter: {CELL_DIAMETER or 'auto'}\n"
            f"flow_threshold: {FLOW_THRESHOLD}\n"
            f"cellprob_threshold: {CELLPROB_THRESHOLD}",
        )

        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        overlay_count = 0
        missing = []
        for img_path in images:
            if create_color_overlay(img_path, output_dir):
                overlay_count += 1
            else:
                missing.append(img_path.name)

        msg = (
            f"Segmentation complete.\n\n"
            f"Results saved to:\n{output_dir}\n\n"
            f"Overlay images created: {overlay_count} / {len(images)}"
        )
        if missing:
            msg += "\n\nNo mask found for:\n" + "\n".join(missing[:10])
        messagebox.showinfo("Finished", msg)

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Cellpose Error", str(e))
    except Exception as e:
        messagebox.showerror("Error", str(e))


def select_folder():
    folder_path = filedialog.askdirectory(title="Select folder containing images")
    if folder_path:
        run_cellpose_on_folder(folder_path)


print("=" * 60)
print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 60)

root = tk.Tk()
root.title("Cellpose GPU Segmentation")
root.geometry("520x300")
root.resizable(False, False)

label = tk.Label(
    root,
    text=(
        "Select a folder of microscopy images\n\n"
        f"Model: {MODEL}   Diameter: {CELL_DIAMETER or 'auto'} px\n"
        f"flow={FLOW_THRESHOLD}   cellprob={CELLPROB_THRESHOLD}\n\n"
        "Edit the parameters at the top of this file\n"
        "if cells are missed or over-merged."
    ),
    font=("Segoe UI", 11),
    justify="center",
)
label.pack(pady=25)

button = tk.Button(
    root,
    text="Select Image Folder",
    command=select_folder,
    width=25,
    height=2,
)
button.pack(pady=10)

root.mainloop()
