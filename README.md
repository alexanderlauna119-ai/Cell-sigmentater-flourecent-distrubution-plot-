# Cell-sigmentater-flourecent-distrubution-plot-
The code was developed using Python 3.12.

Workflow
1. ImageConverterCZItoPNG
   -   Convert CZI microscopy images to PNG format and automatically split the channel containing the cytoskeletal marker into a separate folder.
2. ImageMasksWConda
    - Generate a segmentation mask and ROI for each individual cell using the fluorescence images. By default, the program is configured to run in a Conda environment. If a compatible GPU is        detected, GPU acceleration is used automatically; otherwise, the analysis falls back to CPU processing.
    <img width="927" height="632" alt="image" src="https://github.com/user-attachments/assets/e05ff64c-a85a-4d25-a96f-9463777e3370" />
    <img width="917" height="632" alt="image" src="https://github.com/user-attachments/assets/9da044c5-2a25-4fe4-8056-a230887fc491" />
3. IndividualCellAnalysis (FIJI)
    - Import the generated ROIs into FIJI and quantify each individual cell by measuring fluorescence parameters such as integrated intensity or positive area.
    <img width="923" height="632" alt="image" src="https://github.com/user-attachments/assets/64dddb1c-0797-43d7-9df1-06f7417613fd" />
    <img width="923" height="633" alt="image" src="https://github.com/user-attachments/assets/923b4269-4bde-4523-bef9-7f91f41a4768" />
4. Histogram+Distribution
    - Compile the measurements from all analyzed cells and generate fluorescence distribution plots (histograms) illustrating the fluorescence intensity across the entire experimental dataset.
  <img width="836" height="346" alt="image" src="https://github.com/user-attachments/assets/84ba8306-3d4f-4e5f-86ab-ef89ccb51cdc" />

      

