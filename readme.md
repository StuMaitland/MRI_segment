The MRI Segment application is a tool designed for segmenting MRI images using the SAM2 model. It allows users to
interactively create, edit, and propagate segmentation masks across MRI slices. The application supports various
keyboard shortcuts for efficient mask manipulation and provides functionality to save the segmented masks as NIfTI
files. This tool is particularly useful for medical imaging professionals who need to perform detailed and accurate
segmentations of MRI data.

# Installation Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/StuMaitland/MRI_segment.git
   cd MRI_segment
   ```

2. **Create a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download SAM2 Checkpoints:**
    - Visit the [SAM2 model repository](https://example.com/sam2-checkpoints) to download the required checkpoints.
    - Save the checkpoints to a directory, e.g., `/path/to/checkpoints`.

5. **Run the Application:**
   ```bash
   python load_and_display_nii.py
   ```

6. **Set environment variables**
    ```bash
    export SAM2_CHECKPOINT=/path/to/checkpoints/sam2_hiera_large.pt
    ```

# Using the application

1. **Select mask name**- Press a number(`[1-0]`) key to select a mask slot. Enter the name for the mask e.g. "Right
   anterior compartment"
2. **Draw mask**- `Left click` to select region of interest. `Right click` to select region where target does not exist.
   Refine with as many clicks as needed. If mask is inadequate, reset using `Backspace`
3. **Propagate mask**- Once you're satisfied mask is accurate, press `Enter` to propagate mask to all slices.
4. **Edit mask**- Scroll through slices and refine mask with clicks as needed. Repropagate once complete
5. **Select next mask** using number(`[1-0]`) keys.
6. **Complete segmentation**- Press `Escape` to complete segmentation. This will save the masks to a nii file in the
   same directory as the input file.

# Glossary- Keyboard Shortcuts

- `BackSpace`: Clear the current segment.
- `Escape`: Close all plots, calculate cross-sectional areas, export masks to NIfTI, and quit the application.
- `1-9`: Set the current mask index to the corresponding number.
- `Return`: Propagate masks and clear the current segment.
- `Up/Down` Arrow: Scroll up/down through slices.
