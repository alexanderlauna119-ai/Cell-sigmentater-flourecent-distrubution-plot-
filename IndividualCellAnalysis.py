from ij import IJ
from ij.plugin import ChannelSplitter, ZProjector
from ij.plugin.frame import RoiManager
from ij.measure import ResultsTable
from ij.io import OpenDialog
from ij.process import ImageProcessor

# =================================================
# USER SETTINGS
# =================================================

# Channel indices (0-based)
MJF14_CHANNEL = 1

# Remove Cellpose masks smaller than this
MIN_MASK_AREA = 120

# =================================================
# Clear previous results
# =================================================

IJ.run("Clear Results")

rt = ResultsTable.getResultsTable()
rt.reset()

# =================================================
# Open image
# =================================================

imp = IJ.openImage()

if imp is None:
    IJ.error("No image opened")
    raise SystemExit

# =================================================
# Split channels
# =================================================

channels = ChannelSplitter.split(imp)

if MJF14_CHANNEL >= len(channels):
    IJ.error("MJF14 channel index does not exist")
    raise SystemExit

ch_mjf14 = channels[MJF14_CHANNEL]

# =================================================
# Max projection
# =================================================

ch_mjf14 = ZProjector.run(ch_mjf14, "max")

ch_mjf14_orig = ch_mjf14.duplicate()
ch_mjf14_orig.setTitle("MJF14_MAX")
ch_mjf14_orig.show()

# =================================================
# Open Cellpose mask
# =================================================

od = OpenDialog(
    "Select Cellpose mask (_cp_masks.tif)",
    None
)

maskPath = od.getPath()

if maskPath is None:
    IJ.error("No mask selected")
    raise SystemExit

mask = IJ.openImage(maskPath)
mask.setTitle("Cellpose_Mask")
mask.show()

ip = mask.getProcessor()

# =================================================
# ROI Manager
# =================================================

rm = RoiManager.getInstance()

if rm is None:
    rm = RoiManager()

rm.reset()

# =================================================
# Detect labels
# =================================================

stats = ip.getStatistics()
max_label = int(stats.max)

IJ.log("Max Cellpose label detected: %d" % max_label)

# =================================================
# Create ROIs & filter small masks
# =================================================

roi_labels = []
roi_pixel_areas = []

valid_labels = 0

for label in range(1, max_label + 1):

    ip.setThreshold(
        label,
        label,
        ImageProcessor.NO_LUT_UPDATE
    )

    IJ.run(mask, "Create Selection", "")

    roi = mask.getRoi()

    if roi is None:
        IJ.run(mask, "Select None", "")
        continue

    mask.setRoi(roi)

    # Pixel count from mask ROI
    area_pixels = roi.getStatistics().pixelCount

    if area_pixels < MIN_MASK_AREA:

        IJ.log(
            "Removed label %d (%d pixels)"
            % (label, area_pixels)
        )

        IJ.run(mask, "Select None", "")
        continue

    rm.addRoi(roi)

    roi_labels.append(label)
    roi_pixel_areas.append(area_pixels)

    valid_labels += 1

    IJ.run(mask, "Select None", "")

IJ.log("Valid ROIs created: %d" % valid_labels)

# =================================================
# Measure MJF14
# =================================================

for i in range(rm.getCount()):

    roi = rm.getRoi(i)

    ch_mjf14_orig.setRoi(roi)

    mjf_stats = ch_mjf14_orig.getStatistics()

    rt.incrementCounter()

    row = rt.getCounter() - 1

    rt.setValue(
        "Cellpose_Label",
        row,
        roi_labels[i]
    )

    rt.setValue(
        "Pixel_Area",
        row,
        roi_pixel_areas[i]
    )

    rt.setValue(
        "Mean_MJF14",
        row,
        mjf_stats.mean
    )

    rt.setValue(
        "Integrated_MJF14",
        row,
        mjf_stats.mean * roi_pixel_areas[i]
    )

# =================================================
# Overlay
# =================================================

overlay = ch_mjf14_orig.duplicate()

rm.runCommand(
    overlay,
    "Show All"
)

overlay.setTitle(
    "Cellpose ROIs on MJF14"
)

overlay.show()

# =================================================
# Results
# =================================================

rt.show(
    "MJF14 per Cell"
)

IJ.log("Done")