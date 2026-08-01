import os
import aicspylibczi as czi
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, simpledialog

# --------------------------------------------------------------------------------------------------------
# Create tkinter root (hidden)

root = tk.Tk()
root.withdraw()  # Hide main tkinter window

# --------------------------------------------------------------------------------------------------------
# Select input folder containing CZI files

folder_path = filedialog.askdirectory(title="Select folder containing CZI files")
if not folder_path:
    print("No folder selected. Exiting...")
    exit()

# --------------------------------------------------------------------------------------------------------
# Select output folder for PNGs

output_folder = filedialog.askdirectory(title="Select folder to save converted PNGs")
if not output_folder:
    print("No output folder selected. Exiting...")
    exit()

os.makedirs(output_folder, exist_ok=True)

# --------------------------------------------------------------------------------------------------------
# Ask user which channel to convert (0-based index)

channel_index = simpledialog.askinteger("Channel selection", 
                                        "Enter channel index to convert (0 for first channel, 1 for second, etc.):",
                                        minvalue=0)
if channel_index is None:
    print("No channel selected. Exiting...")
    exit()

# --------------------------------------------------------------------------------------------------------
# Loop over all .czi files and convert

for filename in os.listdir(folder_path):
    if not filename.lower().endswith(".czi"):
        continue  # Skip non-CZI files

    czi_path = os.path.join(folder_path, filename)
    czi_file = czi.CziFile(czi_path)

    try:
        # Read all Z planes for the chosen channel
        z_planes = czi_file.read_mosaic(C=channel_index)

        # Handle 3D Z-stack: use maximum intensity projection
        if z_planes.ndim == 3:
            img_data = np.max(z_planes, axis=0)
        else:
            img_data = z_planes  # <--- fixed typo

        # Normalize to 0-255
        img_data = (255 * (img_data / img_data.max())).astype(np.uint8)

        # Save PNG
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{base_name}_channel{channel_index}.png")
        Image.fromarray(img_data).save(output_path)

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error converting {filename}: {e}")
