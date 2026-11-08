# @File (label="Select CZI Images Folder", style="directory") czi_dir_param
# @File (label="Select Masks Folder", style="directory") mask_dir_param
# @Integer (label="Signal Channel Index (0-based: 0=Ch1, 1=Ch2, 2=Ch3)", value=1) channel_idx
# @Integer (label="Min Mask Area (pixels)", value=120) min_area
# @Boolean (label="Save Individual CSV per Image", value=True) save_ind_csv
# @Boolean (label="Save Combined Master CSV", value=True) save_comb_csv
# @Boolean (label="Save ROI Overlay TIFFs", value=True) save_overlay

"""
================================================================================
Universal Fiji / ImageJ Batch Analysis for CZI + Cellpose Masks
================================================================================
Works in TWO modes seamlessly:
  1. FIJI GUI: Drag and drop this file onto Fiji window, or run via
     Plugins -> Macros -> Run... (A folder picker popup appears).
  2. COMMAND LINE (CMD / PowerShell):
     ImageJ-win64.exe --headless --console --run CellAnalysis_Batch.py "czi_dir_param='C:/czi',mask_dir_param='C:/masks',channel_idx=1,min_area=120"
================================================================================
"""

import os
import sys

from ij import IJ
from ij.plugin import ChannelSplitter, ZProjector
from ij.plugin.frame import RoiManager
from ij.measure import ResultsTable
from ij.io import FileSaver
from ij.process import ImageProcessor


def to_str_path(val):
    """Safely convert File object or string to string path."""
    if val is None:
        return ""
    if hasattr(val, "getAbsolutePath"):
        return str(val.getAbsolutePath())
    return str(val).strip()


def find_matching_mask(czi_stem, mask_directory, ch_idx):
    """
    Search mask directory for a matching mask corresponding to czi_stem.
    Checks common Cellpose naming schemes.
    """
    candidates = [
        os.path.join(mask_directory, "%s_ch%d_cp_masks.tif" % (czi_stem, ch_idx)),
        os.path.join(mask_directory, "%s_ch%d_cp_masks.png" % (czi_stem, ch_idx)),
        os.path.join(mask_directory, "%s_cp_masks.tif" % czi_stem),
        os.path.join(mask_directory, "%s_cp_masks.png" % czi_stem),
        os.path.join(mask_directory, "%s_masks.tif" % czi_stem),
        os.path.join(mask_directory, "%s_masks.png" % czi_stem),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Fallback: fuzzy match on filename
    stem_clean = czi_stem.lower().replace(" ", "").replace("_", "")
    for f in os.listdir(mask_directory):
        f_clean = f.lower().replace(" ", "").replace("_", "")
        if stem_clean in f_clean and ("mask" in f.lower() or "cp" in f.lower()):
            if f.lower().endswith((".tif", ".tiff", ".png")):
                return os.path.join(mask_directory, f)

    return None


def run_batch(czi_dir, mask_dir, ch_idx=1, min_area_px=120, ind_csv=True, comb_csv=True, overlay=True):
    print("=" * 70)
    print("FIJI BATCH CELL QUANTIFICATION (CZI + MASKS)")
    print("=" * 70)
    print("CZI Directory : " + str(czi_dir))
    print("Mask Directory: " + str(mask_dir))
    print("Channel Index : %d" % ch_idx)
    print("Min Cell Area : %d pixels" % min_area_px)
    print("=" * 70)

    if not os.path.isdir(czi_dir):
        msg = "ERROR: CZI folder does not exist:\n" + str(czi_dir)
        print(msg)
        IJ.log(msg)
        return

    if not os.path.isdir(mask_dir):
        msg = "ERROR: Mask folder does not exist:\n" + str(mask_dir)
        print(msg)
        IJ.log(msg)
        return

    czi_files = [
        os.path.join(czi_dir, f)
        for f in sorted(os.listdir(czi_dir))
        if f.lower().endswith(".czi")
    ]

    if not czi_files:
        msg = "ERROR: No .czi files found in folder:\n" + str(czi_dir)
        print(msg)
        IJ.log(msg)
        return

    print("Found %d CZI file(s) to process.\n" % len(czi_files))

    master_table = ResultsTable()
    master_table.reset()

    success_count = 0
    total_cells = 0

    for idx, czi_path in enumerate(czi_files, 1):
        stem = os.path.splitext(os.path.basename(czi_path))[0]
        status_msg = "[%d/%d] Processing: %s" % (idx, len(czi_files), stem)
        print(status_msg)
        IJ.showStatus(status_msg)

        mask_path = find_matching_mask(stem, mask_dir, ch_idx)
        if not mask_path:
            warn = "  -> SKIPPED: Could not find matching mask in %s" % mask_dir
            print(warn)
            IJ.log(warn)
            continue

        print("  -> Matched mask: %s" % os.path.basename(mask_path))

        # 1. Open CZI with Bio-Formats / IJ
        imp = IJ.openImage(czi_path)
        if imp is None:
            err = "  -> ERROR: Failed to open %s" % czi_path
            print(err)
            IJ.log(err)
            continue

        # 2. Split Channels
        channels = ChannelSplitter.split(imp)
        if ch_idx >= len(channels):
            err = "  -> ERROR: Channel %d requested, but image only has %d channel(s)" % (ch_idx, len(channels))
            print(err)
            IJ.log(err)
            imp.close()
            continue

        ch_sig = channels[ch_idx]

        # 3. Maximum Intensity Projection along Z
        if ch_sig.getNSlices() > 1:
            ch_sig_max = ZProjector.run(ch_sig, "max")
        else:
            ch_sig_max = ch_sig.duplicate()

        ch_sig_orig = ch_sig_max.duplicate()

        # Close source channels to conserve RAM
        imp.close()
        for c in channels:
            c.close()
        ch_sig_max.close()

        # 4. Open Cellpose Mask
        mask_imp = IJ.openImage(mask_path)
        if mask_imp is None:
            err = "  -> ERROR: Failed to open mask %s" % mask_path
            print(err)
            IJ.log(err)
            ch_sig_orig.close()
            continue

        ip = mask_imp.getProcessor()
        max_label = int(ip.getStatistics().max)

        # 5. Extract ROIs via ROI Manager
        rm = RoiManager.getInstance()
        if rm is None:
            rm = RoiManager()
        rm.reset()

        roi_labels = []
        roi_areas = []

        for label in range(1, max_label + 1):
            ip.setThreshold(label, label, ImageProcessor.NO_LUT_UPDATE)
            IJ.run(mask_imp, "Create Selection", "")
            roi = mask_imp.getRoi()
            if roi is None:
                IJ.run(mask_imp, "Select None", "")
                continue

            mask_imp.setRoi(roi)
            area_px = roi.getStatistics().pixelCount

            if area_px < min_area_px:
                IJ.run(mask_imp, "Select None", "")
                continue

            rm.addRoi(roi)
            roi_labels.append(label)
            roi_areas.append(area_px)
            IJ.run(mask_imp, "Select None", "")

        valid_count = rm.getCount()
        print("  -> Filtered: %d cells retained (>= %d px)" % (valid_count, min_area_px))

        # 6. Measure Intensity on Channel
        ind_table = ResultsTable()
        ind_table.reset()

        for i in range(valid_count):
            roi = rm.getRoi(i)
            ch_sig_orig.setRoi(roi)
            stats = ch_sig_orig.getStatistics()

            # Per-image table
            ind_table.incrementCounter()
            r = ind_table.getCounter() - 1
            ind_table.setValue("Image", r, stem)
            ind_table.setValue("Cellpose_Label", r, roi_labels[i])
            ind_table.setValue("Pixel_Area", r, roi_areas[i])
            ind_table.setValue("Mean_Intensity", r, stats.mean)
            ind_table.setValue("Integrated_Intensity", r, stats.mean * roi_areas[i])

            # Master table
            master_table.incrementCounter()
            mr = master_table.getCounter() - 1
            master_table.setValue("Image", mr, stem)
            master_table.setValue("Cellpose_Label", mr, roi_labels[i])
            master_table.setValue("Pixel_Area", mr, roi_areas[i])
            master_table.setValue("Mean_Intensity", mr, stats.mean)
            master_table.setValue("Integrated_Intensity", mr, stats.mean * roi_areas[i])

        total_cells += valid_count

        # 7. Save outputs
        if ind_csv:
            csv_path = os.path.join(mask_dir, "%s_ch%d_measurements.csv" % (stem, ch_idx))
            ind_table.save(csv_path)
            print("  -> Saved CSV: %s" % os.path.basename(csv_path))

        if overlay:
            overlay_imp = ch_sig_orig.duplicate()
            rm.runCommand(overlay_imp, "Show All")
            tif_path = os.path.join(mask_dir, "%s_ch%d_overlay.tif" % (stem, ch_idx))
            FileSaver(overlay_imp).saveAsTiff(tif_path)
            overlay_imp.close()
            print("  -> Saved Overlay TIFF: %s" % os.path.basename(tif_path))

        # Clean memory
        mask_imp.close()
        ch_sig_orig.close()
        success_count += 1
        print("")

    # Save Combined Master CSV
    if comb_csv and master_table.getCounter() > 0:
        comb_path = os.path.join(mask_dir, "ALL_CELLS_combined_analysis.csv")
        master_table.save(comb_path)
        print("=" * 70)
        print("MASTER TABLE SAVED:")
        print("  " + comb_path)
        master_table.show("Combined Cell Analysis")

    print("=" * 70)
    print("DONE: Processed %d/%d images (%d total cells measured)." % (success_count, len(czi_files), total_cells))
    print("=" * 70)
    IJ.showStatus("Batch analysis complete.")


# Resolve parameter inputs (works with ImageJ parameter dialog or CLI injection)
raw_czi = globals().get("czi_dir_param", "")
raw_mask = globals().get("mask_dir_param", "")
ch = int(globals().get("channel_idx", 1))
min_px = int(globals().get("min_area", 120))
s_ind = bool(globals().get("save_ind_csv", True))
s_comb = bool(globals().get("save_comb_csv", True))
s_ovl = bool(globals().get("save_overlay", True))

czi_path_resolved = to_str_path(raw_czi)
mask_path_resolved = to_str_path(raw_mask)

# Fallback: CLI sys.argv if invoked directly via jython
if not czi_path_resolved and len(sys.argv) > 1:
    czi_path_resolved = sys.argv[1]
if not mask_path_resolved and len(sys.argv) > 2:
    mask_path_resolved = sys.argv[2]

if czi_path_resolved and mask_path_resolved:
    run_batch(
        czi_dir=czi_path_resolved,
        mask_dir=mask_path_resolved,
        ch_idx=ch,
        min_area_px=min_px,
        ind_csv=s_ind,
        comb_csv=s_comb,
        overlay=s_ovl
    )
else:
    print("No folders provided. Please run inside Fiji or pass folder paths.")
