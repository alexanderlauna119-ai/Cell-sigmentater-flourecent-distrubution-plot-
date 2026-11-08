#!/usr/bin/env python3
"""
CZI to Cellpose Pipeline: Automated Channel Extraction, PNG Conversion & GPU Segmentation.
Combines CZI microscopy channel extraction with automated Cellpose GPU segmentation and
per-cell fluorescence intensity measurement in an easy-to-use GUI.

Runs directly inside your active Conda / PyTorch environment (or dispatches via Conda).
"""

import os
import sys
import json
import shutil
import argparse
import threading
import subprocess
import tempfile
import queue
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for thread-safe overlay generation
import matplotlib.pyplot as plt

# Tkinter imports for GUI (CLI mode remains usable in Python installs without Tk).
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TK_AVAILABLE = True
except ImportError:
    tk = None
    ttk = filedialog = messagebox = None
    TK_AVAILABLE = False

# CZI reader imports
CZI_BACKEND = None
try:
    import czifile
    CZI_BACKEND = "czifile"
except ImportError:
    try:
        from pylibCZIrw import czi as pyczi
        CZI_BACKEND = "pylibCZIrw"
    except ImportError:
        CZI_BACKEND = None


# ==============================================================================
# Basic Presettings & Tuning Notes (from Cellpose GPU Batch Runner)
# ==============================================================================
# Tuning notes:
#   - CELL_DIAMETER: measure a typical cell in pixels. 0 = auto (unreliable on sparse / faint fluorescence).
#   - CELLPROB_THRESHOLD: lower (-1 .. -3) to recover faint cells.
#   - FLOW_THRESHOLD: raise (0.6 .. 0.9) to keep irregular / branched shapes.
#   - CHAN / CHAN2: 0=grayscale, 1=red, 2=green, 3=blue. CHAN2=0 means no separate nuclear channel.
#   - NORM_LOW / NORM_HIGH: percentile contrast stretching (1.0% to 99.0%).

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# ---- Segmentation parameters -------------------------------------------------
MODEL = "cyto3"            # "cyto3" (recommended) / "cyto2" / "cyto" / "nuclei"
CELL_DIAMETER = 60.0       # pixels; set 0.0 for auto-estimate
CHAN = 0                   # signal channel (0 = grayscale for extracted channel)
CHAN2 = 0                  # nuclear channel, 0 = none
FLOW_THRESHOLD = 0.7
CELLPROB_THRESHOLD = -2.0
NORM_LOW, NORM_HIGH = 1.0, 99.0
# -----------------------------------------------------------------------------

# ==============================================================================
# Helper Functions: Conda & Environment Detection
# ==============================================================================

def get_active_conda_env():
    """Return the name of the currently active Conda environment, or None."""
    return os.environ.get("CONDA_DEFAULT_ENV") or None


def find_conda_executable():
    """Locate conda executable across standard installation paths."""
    if "CONDA_EXE" in os.environ and os.path.exists(os.environ["CONDA_EXE"]):
        return os.environ["CONDA_EXE"]

    conda_in_path = shutil.which("conda")
    if conda_in_path:
        return conda_in_path

    user_home = Path.home()
    common_windows_paths = [
        user_home / "anaconda3" / "condabin" / "conda.bat",
        user_home / "miniconda3" / "condabin" / "conda.bat",
        user_home / "anaconda3" / "Scripts" / "conda.exe",
        user_home / "miniconda3" / "Scripts" / "conda.exe",
        Path("C:/ProgramData/anaconda3/condabin/conda.bat"),
        Path("C:/ProgramData/miniconda3/condabin/conda.bat"),
        Path("C:/Users") / user_home.name / "AppData" / "Local" / "anaconda3" / "condabin" / "conda.bat",
        Path("C:/Users") / user_home.name / "AppData" / "Local" / "miniconda3" / "condabin" / "conda.bat",
    ]
    for p in common_windows_paths:
        if p.exists():
            return str(p)

    common_unix_paths = [
        user_home / "anaconda3" / "bin" / "conda",
        user_home / "miniconda3" / "bin" / "conda",
        user_home / "miniforge3" / "bin" / "conda",
        Path("/opt/anaconda3/bin/conda"),
        Path("/opt/miniconda3/bin/conda"),
    ]
    for p in common_unix_paths:
        if p.exists():
            return str(p)

    return None


def run_command_streaming(command, log_callback=None):
    """
    Run an external command and stream its output line by line to console and GUI.
    Handles Windows batch launchers (conda.bat / conda.cmd).
    """
    if os.name == "nt" and command and Path(command[0]).suffix.lower() in {".bat", ".cmd"}:
        import subprocess as _subprocess
        shell_cmd = _subprocess.list2cmdline(command)
        command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", shell_cmd]

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in iter(proc.stdout.readline, ""):
        clean_line = line.rstrip()
        print(clean_line)
        if log_callback:
            log_callback(clean_line)

    proc.stdout.close()
    return proc.wait()


def run_command_captured(command):
    """Run an external command and capture stdout/stderr."""
    if os.name == "nt" and command and Path(command[0]).suffix.lower() in {".bat", ".cmd"}:
        import subprocess as _subprocess
        shell_cmd = _subprocess.list2cmdline(command)
        command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", shell_cmd]
    return subprocess.run(command, capture_output=True, text=True)


# ==============================================================================
# PyTorch & Cellpose Diagnostic Inspector
# ==============================================================================

def inspect_current_environment():
    """
    Check if PyTorch and Cellpose are available directly in sys.executable.
    Returns dict with keys: 'has_torch', 'torch_version', 'cuda_available', 'gpu_name', 'has_cellpose'.
    """
    info = {
        "has_torch": False,
        "torch_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "has_cellpose": False,
        "error": None,
    }
    try:
        import torch
        info["has_torch"] = True
        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["error"] = f"PyTorch check error: {e}"

    try:
        import cellpose
        info["has_cellpose"] = True
    except Exception:
        pass

    return info


def verify_runtime_target(mode: str, conda_env_name: str, use_gpu: bool):
    """
    Verifies that the target Python environment has PyTorch and Cellpose.
    mode is either 'current' (use sys.executable) or 'conda' (use conda run -n <env>).
    Returns (python_prefix, torch_version, cuda_available, gpu_name).
    """
    if mode == "current":
        python_prefix = [sys.executable]
        env_label = get_active_conda_env() or "Current Python"
    else:
        conda_exe = find_conda_executable()
        if not conda_exe:
            raise RuntimeError(
                "Could not find Conda executable on PATH or standard directories.\n"
                "Please run this script inside your Anaconda Prompt, or select 'Use Active Environment'."
            )
        env_label = conda_env_name.strip() or "pytorch"
        python_prefix = [conda_exe, "run", "--no-capture-output", "-n", env_label, "python"]

    # Keep this as a real multi-line Python program. Python's -c accepts
    # compound statements such as try/except only when they begin a new line;
    # joining these statements with semicolons causes a SyntaxError.
    diagnostic_code = "\n".join([
        "import json",
        "res = {'has_torch': False, 'has_cellpose': False, 'torch': None, 'cuda': False, 'gpu': None}",
        "try:",
        "    import torch",
        "    res['has_torch'] = True",
        "    res['torch'] = torch.__version__",
        "    res['cuda'] = bool(torch.cuda.is_available())",
        "    res['gpu'] = torch.cuda.get_device_name(0) if res['cuda'] else None",
        "except Exception as e:",
        "    res['torch_err'] = str(e)",
        "try:",
        "    import cellpose",
        "    res['has_cellpose'] = True",
        "except Exception as e:",
        "    res['cellpose_err'] = str(e)",
        "print(json.dumps(res))",
        "",
    ])

    # Conda on Windows rejects newline-containing arguments to `python -c`.
    # Write the diagnostic to a temporary .py file and pass only its path so
    # both `conda run` and the active interpreter receive a simple argument.
    diagnostic_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_czi_env_check.py", encoding="utf-8", newline="\n", delete=False
        ) as diagnostic_file:
            diagnostic_file.write(diagnostic_code)
            diagnostic_path = Path(diagnostic_file.name)

        check = run_command_captured(python_prefix + [str(diagnostic_path)])
        if check.returncode != 0:
            details = (check.stderr or check.stdout or "").strip()
            if "EnvironmentLocationNotFound" in details:
                active = get_active_conda_env()
                hint = f"Your currently active environment is '{active}'." if active else ""
                raise RuntimeError(
                    f"Conda environment '{env_label}' does not exist on this computer.\n{hint}\n\n"
                    f"Please switch the environment setting to '{active or 'pytorch'}'."
                )
            raise RuntimeError(f"Failed to check environment '{env_label}':\n{details}")
    except OSError as exc:
        raise RuntimeError(f"Could not execute Python in environment '{env_label}': {exc}")
    finally:
        if diagnostic_path is not None:
            try:
                diagnostic_path.unlink(missing_ok=True)
            except OSError:
                pass

    # Parse JSON output from diagnostic
    parsed = None
    for line in reversed(check.stdout.splitlines()):
        try:
            parsed = json.loads(line.strip())
            break
        except json.JSONDecodeError:
            continue

    if not isinstance(parsed, dict) or "has_torch" not in parsed:
        raise RuntimeError(f"Could not read diagnostic output from environment '{env_label}':\n{check.stdout}")

    has_torch = parsed.get("has_torch", False)
    has_cellpose = parsed.get("has_cellpose", False)
    torch_version = parsed.get("torch", "unknown")
    cuda_available = bool(parsed.get("cuda", False))
    gpu_name = parsed.get("gpu") or "CPU"

    if not has_torch:
        raise RuntimeError(
            f"PyTorch is not installed in environment '{env_label}'.\n\n"
            f"Please run in your terminal:\n"
            f"  conda activate {env_label}\n"
            f"  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )

    if not has_cellpose:
        raise RuntimeError(
            f"PyTorch was found, but Cellpose is NOT installed in environment '{env_label}'.\n\n"
            f"Please run in your terminal:\n"
            f"  conda activate {env_label}\n"
            f"  pip install cellpose"
        )

    if use_gpu and not cuda_available:
        raise RuntimeError(
            f"GPU was requested, but PyTorch in environment '{env_label}' cannot access CUDA.\n\n"
            f"Please install a CUDA-enabled PyTorch build and NVIDIA drivers, or uncheck GPU acceleration."
        )

    return python_prefix, torch_version, cuda_available, gpu_name


# ==============================================================================
# Helper Functions: CZI Reading & Channel Extraction
# ==============================================================================

def get_czi_metadata(czi_path: Path):
    """Return channel count and Z-depth from CZI file metadata."""
    if CZI_BACKEND == "czifile":
        with czifile.CziFile(czi_path, squeeze=False) as czi:
            if hasattr(czi, "scenes") and czi.scenes:
                scene = next(iter(czi.scenes.values()))
                axes = tuple(scene.dims)
                shape = tuple(scene.shape)
            elif hasattr(czi, "axes") and hasattr(czi, "shape"):
                axes = tuple(czi.axes)
                shape = tuple(czi.shape)
            else:
                arr = czi.asarray()
                axes = tuple(czi.axes) if hasattr(czi, "axes") else tuple("Y" * arr.ndim)
                shape = arr.shape

            axis_map = dict(zip(axes, shape))
            return {
                "channels": int(axis_map.get("C", 1)),
                "z_slices": int(axis_map.get("Z", 1)),
                "axes": axes,
                "shape": shape,
            }

    elif CZI_BACKEND == "pylibCZIrw":
        from pylibCZIrw import czi as pyczi
        with pyczi.open_czi(czi_path) as czidoc:
            bbox = czidoc.total_bounding_box
            n_channels = bbox.get("C", (0, 1))[1] if isinstance(bbox.get("C"), tuple) else 1
            n_z = bbox.get("Z", (0, 1))[1] if isinstance(bbox.get("Z"), tuple) else 1
            return {"channels": n_channels, "z_slices": n_z, "bbox": bbox}

    else:
        raise ImportError(
            "No CZI reader library installed. Please run in your terminal:\n"
            "  pip install czifile imagecodecs\n"
            "or\n"
            "  pip install pylibCZIrw"
        )


def read_czi_channel(czi_path: Path, channel_idx: int = 1, z_project: str = "max"):
    """Read a CZI channel as a 2D image, projecting Z when needed."""
    if CZI_BACKEND == "czifile":
        with czifile.CziFile(czi_path, squeeze=False) as czi:
            if hasattr(czi, "scenes") and czi.scenes:
                scene = next(iter(czi.scenes.values()))
                axes = tuple(scene.dims)
                data = scene.asarray()
            else:
                axes = tuple(czi.axes) if hasattr(czi, "axes") else ()
                data = czi.asarray()

        if "C" in axes:
            c_axis = axes.index("C")
            n_channels = data.shape[c_axis]
            if not 0 <= channel_idx < n_channels:
                raise ValueError(
                    f"Selected channel {channel_idx}, but {czi_path.name} has "
                    f"{n_channels} channel(s) (indices 0 to {n_channels - 1})."
                )
            data = np.take(data, channel_idx, axis=c_axis)
            axes = axes[:c_axis] + axes[c_axis + 1:]
        elif channel_idx != 0:
            raise ValueError(
                f"{czi_path.name} has no channel dimension; only channel 0 is available."
            )

        if "Z" in axes:
            z_axis = axes.index("Z")
            if data.shape[z_axis] > 1:
                if z_project == "max":
                    data = np.max(data, axis=z_axis)
                else:
                    data = np.mean(data, axis=z_axis)
            else:
                data = np.take(data, 0, axis=z_axis)
            axes = axes[:z_axis] + axes[z_axis + 1:]

        # Select first index in remaining non-spatial dimensions (T, S, etc.)
        for axis in reversed(range(len(axes))):
            dim_name = axes[axis]
            if dim_name not in ("Y", "X"):
                data = np.take(data, 0, axis=axis)
                axes = axes[:axis] + axes[axis + 1:]

        if set(axes) != {"Y", "X"} or data.ndim != 2:
            raise ValueError(
                f"Could not reduce CZI data to a 2D Y/X image. Dimensions: {axes}, shape: {data.shape}."
            )

        if axes == ("X", "Y"):
            data = data.T
        return data

    elif CZI_BACKEND == "pylibCZIrw":
        from pylibCZIrw import czi as pyczi
        with pyczi.open_czi(czi_path) as czidoc:
            bbox = czidoc.total_bounding_box
            z_range = bbox.get("Z", (0, 1))
            num_z = z_range[1] if isinstance(z_range, tuple) else 1

            planes = []
            for z in range(num_z):
                plane = czidoc.read(plane={"C": channel_idx, "Z": z})
                if plane.ndim == 3 and plane.shape[-1] == 1:
                    plane = plane.squeeze(axis=-1)
                planes.append(plane)

            if len(planes) == 1:
                return planes[0]
            stack = np.stack(planes, axis=0)
            if z_project == "max":
                return np.max(stack, axis=0)
            return np.mean(stack, axis=0)

    else:
        raise ImportError("CZI reader missing. Run: pip install czifile imagecodecs")


def convert_channel_to_png(img_2d: np.ndarray, output_png_path: Path, norm_low=NORM_LOW, norm_high=NORM_HIGH):
    """
    Applies percentile contrast stretching and saves image as 8-bit PNG for Cellpose.
    """
    img_float = img_2d.astype(np.float32)
    p_low, p_high = np.percentile(img_float, (norm_low, norm_high))
    if p_high > p_low:
        clipped = np.clip(img_float, p_low, p_high)
        scaled = (clipped - p_low) / (p_high - p_low)
    else:
        scaled = img_float - img_float.min()
        if scaled.max() > 0:
            scaled /= scaled.max()

    img_uint8 = (scaled * 255.0).astype(np.uint8)
    Image.fromarray(img_uint8).save(output_png_path)
    return output_png_path


# ==============================================================================
# Helper Functions: Cellpose Post-Processing & Intensity Extraction
# ==============================================================================

def find_mask(image_path: Path, output_dir: Path):
    """Return the mask file matching this image, or None."""
    candidates = sorted(output_dir.glob(f"{image_path.stem}*_cp_masks*"))
    candidates = [c for c in candidates if c.suffix.lower() in IMAGE_EXTS]
    return candidates[0] if candidates else None


def create_color_overlay(image_path: Path, output_dir: Path):
    """Generate color-tinted cell overlay on top of original grayscale PNG."""
    mask_path = find_mask(image_path, output_dir)
    if not mask_path:
        return None, 0

    img = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))

    n_cells = int(mask.max())
    mask_tint = np.ma.masked_where(mask == 0, mask)

    h, w = img.shape[:2]
    dpi = 200
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.imshow(mask_tint, cmap="nipy_spectral", alpha=0.5, interpolation="nearest")
    ax.axis("off")

    out_path = output_dir / f"{image_path.stem}_overlay.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return out_path, n_cells


def extract_cell_intensities(raw_2d: np.ndarray, mask_path: Path, output_csv_path: Path):
    """
    Measure raw Mean Fluorescence Intensity (MFI) per cell and save to CSV/Excel.
    """
    mask = np.array(Image.open(mask_path))
    n_cells = int(mask.max())
    if n_cells == 0:
        return None

    cell_data = []
    for cell_id in range(1, n_cells + 1):
        cell_pixels = raw_2d[mask == cell_id]
        if len(cell_pixels) > 0:
            cell_data.append({
                "Cell_ID": cell_id,
                "Area_px": int(len(cell_pixels)),
                "Mean_Fluorescence": float(np.mean(cell_pixels)),
                "Median_Fluorescence": float(np.median(cell_pixels)),
                "Integrated_Intensity": float(np.sum(cell_pixels)),
            })

    try:
        import pandas as pd
        df = pd.DataFrame(cell_data)
        excel_path = output_csv_path.with_suffix(".xlsx")
        df.to_excel(excel_path, index=False)
        return excel_path
    except ImportError:
        csv_path = output_csv_path.with_suffix(".csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Cell_ID,Area_px,Mean_Fluorescence,Median_Fluorescence,Integrated_Intensity\n")
            for c in cell_data:
                f.write(
                    f"{c['Cell_ID']},{c['Area_px']},"
                    f"{c['Mean_Fluorescence']:.2f},{c['Median_Fluorescence']:.2f},"
                    f"{c['Integrated_Intensity']:.2f}\n"
                )
        return csv_path


# ==============================================================================
# Pipeline Worker Engine (Called by GUI or CLI)
# ==============================================================================

def execute_pipeline(
    input_path: Path,
    channel_idx: int = 1,
    z_project: str = "max",
    env_mode: str = "current",
    conda_env_name: str = "pytorch",
    model: str = MODEL,
    diameter: float = CELL_DIAMETER,
    flow_threshold: float = FLOW_THRESHOLD,
    cellprob_threshold: float = CELLPROB_THRESHOLD,
    chan: int = CHAN,
    chan2: int = CHAN2,
    norm_low: float = NORM_LOW,
    norm_high: float = NORM_HIGH,
    use_gpu: bool = True,
    log_callback=None,
):
    """
    Core pipeline: extracts CZI channels -> PNGs -> runs Cellpose -> overlays & tables.
    """
    def log(msg):
        print(msg)
        if log_callback:
            log_callback(msg)

    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    is_folder = input_path.is_dir()
    if is_folder:
        czi_list = sorted(list(input_path.glob("*.czi")))
        if not czi_list:
            raise ValueError(f"No .czi files found in folder: {input_path}")
        base_dir = input_path
    else:
        czi_list = [input_path]
        base_dir = input_path.parent

    # Channel-scoped folders ensure different channels never mix or re-segment each other
    output_dir = base_dir / f"cellpose_output_ch{channel_idx}"
    png_dir = base_dir / f"extracted_channels_ch{channel_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 65)
    log(f"Starting pipeline for {len(czi_list)} CZI file(s)")
    log(f"Channel: {channel_idx} | Z-projection: {z_project} | Model: {model} ({diameter}px)")
    log(f"Presets: flow={flow_threshold} | cellprob={cellprob_threshold} | norm=({norm_low}%, {norm_high}%) | chan={chan}, chan2={chan2}")
    log(f"Extracted PNG folder: {png_dir.resolve()}")
    log(f"Cellpose output folder: {output_dir.resolve()}")
    log("=" * 65)

    # 1. Verify runtime environment
    log("\n[1/3] Verifying Python, PyTorch & Cellpose environment...")
    python_prefix, torch_version, cuda_available, gpu_name = verify_runtime_target(
        mode=env_mode, conda_env_name=conda_env_name, use_gpu=use_gpu
    )
    log(f"  ✓ PyTorch {torch_version} verified")
    log(f"  ✓ CUDA Available: {cuda_available} ({gpu_name})")

    # 2. Extract CZI channels to normalized PNGs
    log("\n[2/3] Extracting Channel to normalized 8-bit PNG...")
    extracted_pngs = []
    raw_arrays = {}

    for idx, czi_file in enumerate(czi_list, 1):
        log(f"  [{idx}/{len(czi_list)}] Reading {czi_file.name}...")
        raw_2d = read_czi_channel(czi_file, channel_idx=channel_idx, z_project=z_project)
        png_name = f"{czi_file.stem}_ch{channel_idx}.png"
        png_path = png_dir / png_name
        convert_channel_to_png(raw_2d, png_path, norm_low=norm_low, norm_high=norm_high)
        if not png_path.is_file() or png_path.stat().st_size == 0:
            raise IOError(f"PNG was not written or is empty: {png_path.resolve()}")
        extracted_pngs.append(png_path)
        raw_arrays[png_path.stem] = raw_2d
        log(f"      -> Saved PNG: {png_path.resolve()} (Dimensions: {raw_2d.shape[1]}x{raw_2d.shape[0]})")

    # 3. Build & execute Cellpose command
    log(f"\n[3/3] Running Cellpose GPU Segmentation on extracted Channel {channel_idx}...")
    log(f"  Note: Cellpose logs 'using all channels' because --chan 0 is Cellpose's command-line")
    log(f"  flag for single-channel grayscale images. Only Channel {channel_idx} is present in the input.")
    cellpose_args = [
        "-m", "cellpose",
        "--dir", str(png_dir),
        "--pretrained_model", model,
        "--diameter", str(diameter),
        "--chan", str(chan),
        "--chan2", str(chan2),
        "--flow_threshold", str(flow_threshold),
        "--cellprob_threshold", str(cellprob_threshold),
        "--norm_percentile", str(norm_low), str(norm_high),
        "--save_png",
        "--save_tif",
        "--verbose",
        "--savedir", str(output_dir),
    ]
    if use_gpu and cuda_available:
        cellpose_args.append("--use_gpu")

    full_cmd = python_prefix + cellpose_args
    log(f"Command: {' '.join(full_cmd)}\n")

    return_code = run_command_streaming(full_cmd, log_callback=log)
    if return_code != 0:
        raise RuntimeError(f"Cellpose exited with error code {return_code}.")

    # 4. Generate Overlays & Extract Intensity Measurements
    log("\nGenerating color overlays and measuring cell intensities...")
    total_cells = 0
    for png_path in extracted_pngs:
        overlay_path, n_cells = create_color_overlay(png_path, output_dir)
        total_cells += n_cells
        if overlay_path:
            log(f"  ✓ {png_path.name}: {n_cells} cells segmented -> {overlay_path.resolve()}")
        mask_file = find_mask(png_path, output_dir)
        if mask_file:
            log(f"      -> Cellpose mask: {mask_file.resolve()}")
        else:
            log(f"  ! No Cellpose mask found for {png_path.name} in {output_dir.resolve()}")
        if mask_file and png_path.stem in raw_arrays:
            data_file = extract_cell_intensities(
                raw_arrays[png_path.stem],
                mask_file,
                output_dir / f"{png_path.stem}_cell_intensities",
            )
            if data_file:
                log(f"      -> Cell intensity table: {data_file.resolve()}")

    log("\n" + "=" * 65)
    log(f"PIPELINE COMPLETE!")
    log(f"Images processed : {len(extracted_pngs)}")
    log(f"Total cells found: {total_cells}")
    log(f"Extracted channel PNGs: {png_dir.resolve()}")
    log(f"Masks, overlays & tables: {output_dir.resolve()}")
    png_files = sorted(png_dir.glob("*.png"))
    result_files = sorted(p for p in output_dir.iterdir() if p.is_file())
    log(f"Verified files on disk: {len(png_files)} PNG(s) in extracted folder; {len(result_files)} file(s) in output folder")
    if not png_files:
        log("WARNING: No PNG files are currently visible in the extracted folder.")
    if not result_files:
        log("WARNING: Cellpose output folder appears empty on disk.")
    log("=" * 65)

    return len(extracted_pngs), total_cells, output_dir


# ==============================================================================
# Modern Tkinter GUI Application
# ==============================================================================

class CZICellposeGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("CZI to Cellpose Pipeline (Auto PyTorch & GPU)")
        self.root.geometry("680x760")
        self.root.minsize(620, 680)

        # State variables
        self.path_var = tk.StringVar()
        self.is_folder_var = tk.BooleanVar(value=False)
        self.channel_var = tk.IntVar(value=1)
        self.z_proj_var = tk.StringVar(value="max")

        # Cellpose parameters (initialized from presets)
        self.model_var = tk.StringVar(value=MODEL)
        self.diameter_var = tk.DoubleVar(value=CELL_DIAMETER)
        self.flow_thresh_var = tk.DoubleVar(value=FLOW_THRESHOLD)
        self.cellprob_thresh_var = tk.DoubleVar(value=CELLPROB_THRESHOLD)
        self.chan_var = tk.IntVar(value=CHAN)
        self.chan2_var = tk.IntVar(value=CHAN2)
        self.norm_low_var = tk.DoubleVar(value=NORM_LOW)
        self.norm_high_var = tk.DoubleVar(value=NORM_HIGH)
        self.use_gpu_var = tk.BooleanVar(value=True)

        # Environment detection
        self.active_env = get_active_conda_env()
        self.default_target_env = self.active_env or "pytorch"
        # Prefer the already-active env; outside Conda, dispatch to the pytorch env.
        self.default_env_mode = "current" if self.active_env else "conda"
        self.env_mode_var = tk.StringVar(value=self.default_env_mode)
        self.conda_env_name_var = tk.StringVar(value=self.default_target_env)
        self.ui_queue = queue.Queue()

        self._build_ui()
        self._detect_environment_hardware()
        self.root.after(100, self._poll_ui_events)

    def _build_ui(self):
        # Configure ttk style
        style = ttk.Style()
        style.theme_use("clam")

        main_frame = ttk.Frame(self.root, padding="14")
        main_frame.pack(fill="both", expand=True)

        # Title
        title_box = ttk.Frame(main_frame)
        title_box.pack(fill="x", pady=(0, 8))
        ttk.Label(
            title_box,
            text="Zeiss CZI → Channel Extraction → Cellpose GPU",
            font=("Segoe UI", 13, "bold"),
            foreground="#0369a1",
        ).pack(side="left")

        # Hardware badge
        self.hw_label = ttk.Label(
            title_box,
            text="Checking PyTorch & CUDA...",
            font=("Segoe UI", 9, "bold"),
            foreground="#475569",
        )
        self.hw_label.pack(side="right")

        # --- Section 1: Input Zeiss File / Folder ---
        sec1 = ttk.LabelFrame(main_frame, text=" 1. Select Zeiss CZI File or Folder ", padding="10")
        sec1.pack(fill="x", pady=5)

        path_box = ttk.Frame(sec1)
        path_box.pack(fill="x")
        ttk.Entry(path_box, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(path_box, text="Browse File (.czi)", command=self._browse_file).pack(side="left", padx=2)
        ttk.Button(path_box, text="Browse Folder", command=self._browse_folder).pack(side="left", padx=2)

        self.info_lbl = ttk.Label(
            sec1,
            text="Browse a .czi file to inspect channels and Z-slices.",
            foreground="#64748b",
        )
        self.info_lbl.pack(anchor="w", pady=(6, 0))

        # --- Section 2: Channel & Conversion Settings ---
        sec2 = ttk.LabelFrame(main_frame, text=" 2. Channel & Image Settings ", padding="10")
        sec2.pack(fill="x", pady=5)

        ch_frame = ttk.Frame(sec2)
        ch_frame.pack(fill="x")

        ttk.Label(ch_frame, text="Target Channel Index:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.channel_spin = ttk.Spinbox(ch_frame, from_=0, to=10, textvariable=self.channel_var, width=5)
        self.channel_spin.grid(row=0, column=1, sticky="w", padx=4, pady=3)
        ttk.Label(ch_frame, text="(e.g., 0 = DAPI, 1 = Signal, 2 = Marker)", foreground="#64748b").grid(row=0, column=2, sticky="w", padx=8)

        ttk.Label(ch_frame, text="Z-Stack Projection:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        ttk.Combobox(ch_frame, textvariable=self.z_proj_var, values=["max", "mean"], state="readonly", width=8).grid(row=1, column=1, sticky="w", padx=4, pady=3)

        # --- Section 3: Cellpose Parameters ---
        sec3 = ttk.LabelFrame(main_frame, text=" 3. Cellpose GPU Parameters (Default Presets) ", padding="10")
        sec3.pack(fill="x", pady=5)

        cp_frame = ttk.Frame(sec3)
        cp_frame.pack(fill="x")

        ttk.Label(cp_frame, text="Model:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Combobox(cp_frame, textvariable=self.model_var, values=["cyto3", "cyto2", "cyto", "nuclei"], state="readonly", width=10).grid(row=0, column=1, sticky="w", padx=4, pady=3)

        ttk.Label(cp_frame, text="Cell Diameter (px):").grid(row=0, column=2, sticky="w", padx=(18, 4), pady=3)
        ttk.Entry(cp_frame, textvariable=self.diameter_var, width=8).grid(row=0, column=3, sticky="w", padx=4, pady=3)

        ttk.Label(cp_frame, text="Flow Threshold:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(cp_frame, textvariable=self.flow_thresh_var, width=8).grid(row=1, column=1, sticky="w", padx=4, pady=3)

        ttk.Label(cp_frame, text="Cellprob Threshold:").grid(row=1, column=2, sticky="w", padx=(18, 4), pady=3)
        ttk.Entry(cp_frame, textvariable=self.cellprob_thresh_var, width=8).grid(row=1, column=3, sticky="w", padx=4, pady=3)

        ttk.Label(cp_frame, text="Norm Low %:").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(cp_frame, textvariable=self.norm_low_var, width=8).grid(row=2, column=1, sticky="w", padx=4, pady=3)

        ttk.Label(cp_frame, text="Norm High %:").grid(row=2, column=2, sticky="w", padx=(18, 4), pady=3)
        ttk.Entry(cp_frame, textvariable=self.norm_high_var, width=8).grid(row=2, column=3, sticky="w", padx=4, pady=3)

        row3 = ttk.Frame(cp_frame)
        row3.grid(row=3, column=0, columnspan=4, sticky="ew", padx=4, pady=(6, 2))
        ttk.Checkbutton(row3, text="Use GPU Acceleration (CUDA)", variable=self.use_gpu_var).pack(side="left")
        ttk.Button(row3, text="↺ Reset to Presets", command=self._reset_presets).pack(side="right")

        notes_lbl = ttk.Label(
            cp_frame,
            text="Presets: cyto3 | diameter=60px (0=auto) | flow=0.7 (raise to 0.8-0.9 for branched) | cellprob=-2.0 (down to -3 for faint) | norm=1.0% to 99.0%",
            font=("Segoe UI", 8),
            foreground="#64748b",
        )
        notes_lbl.grid(row=4, column=0, columnspan=4, sticky="w", padx=4, pady=(3, 0))

        # --- Section 4: Python & Conda Environment ---
        sec4 = ttk.LabelFrame(main_frame, text=" 4. Python / Conda Environment ", padding="10")
        sec4.pack(fill="x", pady=5)

        env_frame = ttk.Frame(sec4)
        env_frame.pack(fill="x")

        r1 = ttk.Radiobutton(
            env_frame,
            text=(f"Use Active Environment ('{self.active_env}')" if self.active_env
                  else "Use Active Environment (launch from the desired Conda environment)"),
            variable=self.env_mode_var,
            value="current",
        )
        r1.pack(anchor="w", pady=(0, 2))

        r2_box = ttk.Frame(env_frame)
        r2_box.pack(fill="x", pady=(2, 0))
        r2 = ttk.Radiobutton(
            r2_box,
            text="Or run via Conda environment name:",
            variable=self.env_mode_var,
            value="conda",
        )
        r2.pack(side="left")
        ttk.Entry(r2_box, textvariable=self.conda_env_name_var, width=14).pack(side="left", padx=6)
        ttk.Label(r2_box, text="(e.g. pytorch)", foreground="#64748b").pack(side="left", padx=4)

        # --- Action Button & Console Output ---
        self.run_btn = tk.Button(
            main_frame,
            text="▶ Extract Channel & Run Cellpose GPU",
            command=self._start_pipeline,
            bg="#0284c7",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            pady=8,
            cursor="hand2",
        )
        self.run_btn.pack(fill="x", pady=10)

        # Log window
        log_frame = ttk.LabelFrame(main_frame, text=" Progress & Output Console ", padding="6")
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, height=9, font=("Consolas", 9), wrap="word", bg="#0f172a", fg="#f8fafc")
        self.log_text.pack(fill="both", expand=True)

    def _detect_environment_hardware(self):
        """Inspect PyTorch and CUDA in the active Python environment and update badge."""
        info = inspect_current_environment()
        if info["has_torch"]:
            if info["cuda_available"]:
                gpu = info["gpu_name"] or "NVIDIA GPU"
                self.hw_label.config(text=f"● PyTorch {info['torch_version']} | CUDA Active ({gpu})", foreground="#15803d")
            else:
                self.hw_label.config(text=f"● PyTorch {info['torch_version']} | CPU Mode", foreground="#b45309")
        else:
            self.hw_label.config(text="● PyTorch not in current environment (will use Conda)", foreground="#64748b")

    def _log(self, text: str):
        """Queue worker-thread messages; Tk widgets are updated only on the UI thread."""
        self.ui_queue.put(("log", text))

    def _poll_ui_events(self):
        """Safely apply worker updates on Tk's main thread."""
        try:
            while True:
                event, payload = self.ui_queue.get_nowait()
                if event == "log":
                    self.log_text.insert("end", payload + "\n")
                    self.log_text.see("end")
                elif event == "success":
                    n_files, total_cells, out_dir = payload
                    self.run_btn.config(state="normal", text="▶ Extract Channel & Run Cellpose GPU", bg="#0284c7")
                    messagebox.showinfo(
                        "Success",
                        f"Segmentation Pipeline Complete!\n\n"
                        f"• Images processed: {n_files}\n"
                        f"• Total cells segmented: {total_cells}\n\n"
                        f"Results saved to:\n{out_dir}",
                    )
                elif event == "error":
                    self.run_btn.config(state="normal", text="▶ Extract Channel & Run Cellpose GPU", bg="#0284c7")
                    self.log_text.insert("end", "\n[!] ERROR: " + payload + "\n")
                    self.log_text.see("end")
                    messagebox.showerror("Pipeline Error", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_ui_events)

    def _browse_file(self):
        f = filedialog.askopenfilename(
            title="Select CZI Microscope Image",
            filetypes=[("Zeiss CZI Files", "*.czi"), ("All Files", "*.*")],
        )
        if f:
            self.path_var.set(f)
            self.is_folder_var.set(False)
            self._inspect_path(Path(f))

    def _browse_folder(self):
        d = filedialog.askdirectory(title="Select Folder containing CZI Files")
        if d:
            self.path_var.set(d)
            self.is_folder_var.set(True)
            self._inspect_path(Path(d))

    def _inspect_path(self, path: Path):
        try:
            if path.is_file():
                meta = get_czi_metadata(path)
                n_ch = meta["channels"]
                n_z = meta["z_slices"]
                self.info_lbl.config(
                    text=f"Detected: {path.name} | Channels: {n_ch} (indices 0 to {n_ch-1}) | Z-slices: {n_z}",
                    foreground="#047857",
                )
                self.channel_spin.config(to=max(0, n_ch - 1))
            elif path.is_dir():
                czi_files = list(path.glob("*.czi"))
                if czi_files:
                    meta = get_czi_metadata(czi_files[0])
                    n_ch = meta["channels"]
                    n_z = meta["z_slices"]
                    self.info_lbl.config(
                        text=f"Folder: {len(czi_files)} .czi file(s) found. Channels: {n_ch} | Z-slices: {n_z}",
                        foreground="#047857",
                    )
                    self.channel_spin.config(to=max(0, n_ch - 1))
                else:
                    self.info_lbl.config(text=f"No .czi files found in folder: {path.name}", foreground="#b91c1c")
        except Exception as e:
            self.info_lbl.config(text=f"Notice: {e}", foreground="#b91c1c")

    def _reset_presets(self):
        """Restore all parameters to the exact preset defaults."""
        self.model_var.set(MODEL)
        self.diameter_var.set(CELL_DIAMETER)
        self.flow_thresh_var.set(FLOW_THRESHOLD)
        self.cellprob_thresh_var.set(CELLPROB_THRESHOLD)
        self.chan_var.set(CHAN)
        self.chan2_var.set(CHAN2)
        self.norm_low_var.set(NORM_LOW)
        self.norm_high_var.set(NORM_HIGH)
        self.use_gpu_var.set(True)

    def _start_pipeline(self):
        p_str = self.path_var.get().strip().strip('"').strip("'")
        if not p_str:
            messagebox.showwarning("Missing Input", "Please select a .czi file or folder first.")
            return

        target_path = Path(p_str)
        if not target_path.exists():
            messagebox.showerror("Error", f"Path does not exist:\n{target_path}")
            return

        self.run_btn.config(state="disabled", text="Processing... Please wait", bg="#64748b")
        self.log_text.delete("1.0", "end")

        # Snapshot Tk values on the UI thread, then work only with plain Python values.
        options = {
            "channel_idx": self.channel_var.get(),
            "z_project": self.z_proj_var.get(),
            "env_mode": self.env_mode_var.get(),
            "conda_env_name": self.conda_env_name_var.get().strip() or "pytorch",
            "model": self.model_var.get(),
            "diameter": self.diameter_var.get(),
            "flow_threshold": self.flow_thresh_var.get(),
            "cellprob_threshold": self.cellprob_thresh_var.get(),
            "chan": self.chan_var.get(),
            "chan2": self.chan2_var.get(),
            "norm_low": self.norm_low_var.get(),
            "norm_high": self.norm_high_var.get(),
            "use_gpu": self.use_gpu_var.get(),
        }
        threading.Thread(target=self._run_worker, args=(target_path, options), daemon=True).start()

    def _run_worker(self, target_path: Path, options: dict):
        try:
            n_files, total_cells, out_dir = execute_pipeline(
                input_path=target_path,
                **options,
                log_callback=self._log,
            )

            self.ui_queue.put(("success", (n_files, total_cells, out_dir)))
        except Exception as e:
            self.ui_queue.put(("error", str(e)))


# ==============================================================================
# Main Entry Point: CLI or GUI
# ==============================================================================

def main():
    active_env = get_active_conda_env()
    default_env = active_env or "pytorch"

    parser = argparse.ArgumentParser(
        description="Zeiss CZI to Cellpose GPU Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input", type=str, default=None, help="Path to .czi file or folder")
    parser.add_argument("-c", "--channel", type=int, default=1, help="Channel index (default: 1)")
    parser.add_argument("-z", "--z-project", type=str, choices=["max", "mean"], default="max", help="Z projection")
    parser.add_argument("--env", type=str, default=default_env, help=f"Conda env (default: {default_env})")
    parser.add_argument("--no-conda", action="store_true", help="Force using current Python directly")
    parser.add_argument("--model", type=str, default=MODEL, help=f"Cellpose model (default: {MODEL})")
    parser.add_argument("--diameter", type=float, default=CELL_DIAMETER, help=f"Cell diameter px (default: {CELL_DIAMETER})")
    parser.add_argument("--flow-threshold", type=float, default=FLOW_THRESHOLD, help=f"Flow threshold (default: {FLOW_THRESHOLD})")
    parser.add_argument("--cellprob-threshold", type=float, default=CELLPROB_THRESHOLD, help=f"Cellprob threshold (default: {CELLPROB_THRESHOLD})")
    parser.add_argument("--norm-low", type=float, default=NORM_LOW, help=f"Norm low percentile (default: {NORM_LOW})")
    parser.add_argument("--norm-high", type=float, default=NORM_HIGH, help=f"Norm high percentile (default: {NORM_HIGH})")
    parser.add_argument("--chan", type=int, default=CHAN, help=f"Signal channel (default: {CHAN})")
    parser.add_argument("--chan2", type=int, default=CHAN2, help=f"Nuclear channel (default: {CHAN2})")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU (use CPU)")

    args = parser.parse_args()

    # If run with CLI input flag, execute directly in terminal
    if args.input:
        env_mode = "current" if (args.no_conda or (active_env and args.env.lower() == active_env.lower())) else "conda"
        execute_pipeline(
            input_path=Path(args.input.strip().strip('"').strip("'")),
            channel_idx=args.channel,
            z_project=args.z_project,
            env_mode=env_mode,
            conda_env_name=args.env,
            model=args.model,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            chan=args.chan,
            chan2=args.chan2,
            norm_low=args.norm_low,
            norm_high=args.norm_high,
            use_gpu=not args.no_gpu,
        )
    else:
        # Otherwise, launch the full GUI
        if not TK_AVAILABLE:
            raise RuntimeError(
                "Tkinter is unavailable in this Python installation, so the GUI cannot start. "
                "On Windows, run this script from Anaconda Prompt using your pytorch environment."
            )
        root = tk.Tk()
        app = CZICellposeGUI(root)
        root.mainloop()


if __name__ == "__main__":
    main()
