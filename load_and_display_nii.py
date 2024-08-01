import tkinter as tk
from tkinter import filedialog
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

import segment
from segment import segment_image, setup_segment


class InteractiveSegment():
    file_path = str()

    img_data = np.array([])
    slice_index = 0
    input_point = np.empty((0, 2), int)
    input_label = np.empty((0,), int)
    current_mask = np.array([])
    saved_masks = {}

    predictor = setup_segment()

    def __init__(self):
        self.file_path = self.select_nii_file()

    def select_nii_file(self):
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename(
            filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")]
        )
        return file_path

    def load_nii_file(self):
        # Load the .nii file
        img = nib.load(self.file_path)
        # Get the image data as a numpy array
        self.img_data = img.get_fdata()
        # img_data = np.flipud(np.fliplr(img_data))
        # img_data = np.rot90(img_data, k=-1, axes=(0, 1))

    def update_image(self, ax, canvas):
        ax.clear()
        ax.imshow(self.img_data[:, :, self.slice_index], cmap='bone')
        if self.current_mask.size > 0:
            mask_overlay = np.ma.masked_where(self.current_mask == 0, self.current_mask)
            ax.imshow(mask_overlay, cmap='jet', alpha=0.5)

        ax.axis('off')
        canvas.draw()

    def clear_segment(self):
        self.current_mask = np.array([])
        self.input_point = np.empty((0, 2), int)
        self.input_label = np.empty((0,), int)

    def convert_to_rgb(self, grayscale_slice):
        # Normalize the grayscale values to the range [0, 1]
        normalized_slice = (grayscale_slice - np.min(grayscale_slice)) / (
                np.max(grayscale_slice) - np.min(grayscale_slice))

        # Stack the normalized grayscale values along the third dimension to create an RGB image
        rgb_image = np.stack((normalized_slice,) * 3, axis=-1)
        rgb_image = rgb_image.astype(np.float32)  # Convert to float32
        return rgb_image

    def all_cross_sectional_areas(self, pixel_area=1.0):
        """
        Calculate the cross-sectional areas for all saved masks.

        Parameters:
        pixel_area (float): The area of a single pixel. Default is 1.0.

        Returns:
        """
        for key, mask_info in self.saved_masks.items():
            mask = mask_info['mask']
            mask_name = mask_info['name']
            binary_mask = mask > 0

            # Count the number of non-zero pixels
            num_non_zero_pixels = np.sum(binary_mask)

            # Calculate the cross-sectional area
            cross_sectional_area = num_non_zero_pixels * pixel_area
            self.saved_masks['key']['area'] = cross_sectional_area

    def on_scroll(self, event, ax, canvas):
        if event.delta > 0:
            self.slice_index = min(self.slice_index + 1, self.img_data.shape[2] - 1)
        else:
            self.slice_index = max(self.slice_index - 1, 0)
        self.update_image(ax, canvas)

    def on_click(self, event, ax, canvas):
        print(f' button: {event.button}')
        x, y = int(event.xdata), int(event.ydata)
        self.input_point = np.append(self.input_point, [[x, y]], axis=0)
        if event.button == 1:  # Left click
            self.input_label = np.append(self.input_label, 1)
        elif event.button == 3:  # Right click
            self.input_label = np.append(self.input_label, 0)
        rgb_image = self.convert_to_rgb(self.img_data[:, :, self.slice_index])
        self.current_mask = segment_image(self.predictor, rgb_image, self.input_point, self.input_label, self.current_mask)
        self.update_image(ax, canvas)
        canvas.draw()

    def on_keypress(self, event):
        if event.keysym == 'BackSpace':
            self.clear_segment()
        elif event.char.isdigit() and 1 <= int(event.char) <= 9:
            mask_name = tk.simpledialog.askstring("Input", "Enter mask name:")

            key = int(event.char)

            if key not in self.saved_masks:
                self.saved_masks[key] = {'mask': None, 'name': None}

            self.saved_masks[key]['mask'] = self.current_mask
            self.saved_masks[key]['name'] = mask_name
            self.clear_segment()


# Example usage
if __name__ == "__main__":
    interactive_segment = InteractiveSegment()
    interactive_segment.load_nii_file()

    root = tk.Tk()
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    interactive_segment.update_image(ax, canvas)

    root.bind("<MouseWheel>", lambda event: interactive_segment.on_scroll(event, ax, canvas))
    cid = canvas.mpl_connect("button_release_event", lambda event: interactive_segment.on_click(event, ax, canvas))
    root.bind("<KeyPress>", lambda event: interactive_segment.on_keypress(event))

    print(f"Event connection ID: {cid}")
    root.mainloop()

# Pseudocode for next steps:
# Left click- add point to array, start segmentation
# Right click- add a not-point, start segmentation
# Press delete key- remove last point
# Press number key- save most recent mask as that number
# Press enter- save output of all masks
