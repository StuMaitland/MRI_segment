import tkinter as tk
from tkinter import filedialog
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os

import segment
from segment import segment_image, setup_segment


class InteractiveSegment():
    file_path = str()

    img_data = np.array([])
    slice_index = 0
    input_point = np.empty((0, 2), int)
    input_label = np.empty((0,), int)
    current_masks = []  # List to store multiple masks
    current_logits = []  # List to store multiple logits
    mask_names = {}
    current_mask_index = 1
    current_display_mask_index = 0  # Index to track the currently displayed mask

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
        img_data = np.flipud(np.fliplr(self.img_data))
        self.img_data = np.rot90(img_data, k=-1, axes=(0, 1))

    def update_image(self, ax, canvas):
        ax.clear()
        ax.imshow(self.img_data[:, :, self.slice_index], cmap='bone')

        # Define a list of colors for the masks
        colors = [
            [1, 0, 0, 0.5],  # Red
            [0, 1, 0, 0.5],  # Green
            [0, 0, 1, 0.5],  # Blue
            [1, 1, 0, 0.5],  # Yellow
            [1, 0, 1, 0.5],  # Magenta
            [0, 1, 1, 0.5],  # Cyan
            [0.5, 0.5, 0.5, 0.5],  # Gray
            [1, 0.5, 0, 0.5],  # Orange
            [0.5, 0, 1, 0.5],  # Purple
        ]

        # Overlay all saved masks with different colors
        for i, (key, mask_info) in enumerate(self.mask_names.items()):
            if 'mask' in mask_info and self.slice_index in mask_info['mask']:
                mask = mask_info['mask'][self.slice_index]
                if mask is not None and mask.size > 0:
                    color = colors[i % len(colors)]
                    mask_overlay = np.ma.masked_where(mask == 0, mask)
                    cmap = plt.cm.colors.ListedColormap([color])
                    ax.imshow(mask_overlay, cmap=cmap, alpha=0.5)

        # Overlay the current mask with a different color
        if len(self.current_masks) > 0:
            current_mask = self.current_masks[self.current_display_mask_index]
            if current_mask.size > 0:
                mask_overlay = np.ma.masked_where(current_mask == 0, current_mask)
                ax.imshow(mask_overlay, cmap='jet', alpha=0.5)

        ax.axis('off')
        canvas.draw()

    def clear_segment(self):
        self.current_masks = []
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
        for key, mask_info in self.mask_names.items():
            total_area = 0
            slice_count = 0
            for slice_index, mask in mask_info['mask'].items():
                binary_mask = mask > 0

                # Count the number of non-zero pixels
                num_non_zero_pixels = np.sum(binary_mask)

                # Calculate the cross-sectional area for this slice
                cross_sectional_area = num_non_zero_pixels * pixel_area
                total_area += cross_sectional_area
                slice_count += 1

            # Calculate the mean cross-sectional area for this mask
            mean_area = total_area / slice_count if slice_count > 0 else 0
            self.mask_names[key]['area'] = mean_area

    def export_masks_to_nifti(self, output_dir):
        for key, mask_info in self.mask_names.items():
            # Create a 3D array to store the mask across all slices
            mask_3d = np.zeros(self.img_data.shape, dtype=np.uint8)

            for slice_index, mask in mask_info['mask'].items():
                mask_3d[:, :, slice_index] = mask

            # Create a NIfTI image from the 3D mask array
            nifti_img = nib.Nifti1Image(mask_3d, np.eye(4))

            # Save the NIfTI image to a file
            mask_name = mask_info.get('name', f'mask_{key}')
            output_path = f"{output_dir}/{mask_name}.nii"
            nib.save(nifti_img, output_path)

    def set_mask_index(self, index):
        self.current_mask_index = index
        if index not in self.mask_names:
            mask_name = tk.simpledialog.askstring("Input", "Enter mask name:")
            self.mask_names[index] = {'mask': {}, 'name': mask_name}

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
        self.current_masks, self.current_logits = segment_image(self.predictor, rgb_image, self.input_point,
                                                                self.input_label, self.current_logits)

        self.update_image(ax, canvas)

    def on_keypress(self, event):
        if event.keysym == 'BackSpace':
            self.clear_segment()
            self.update_image(ax, canvas)
        elif event.keysym == 'Escape':
            plt.close('all')
            self.all_cross_sectional_areas()
            self.export_masks_to_nifti(os.path.dirname(self.file_path))
            root.quit()
        elif event.char.isdigit() and 1 <= int(event.char) <= 9:
            key = int(event.char)
            self.set_mask_index(key)
        elif event.keysym == 'Return':
            self.set_mask_index(self.current_mask_index)
            if 'mask' not in self.mask_names[self.current_mask_index]:
                self.mask_names[self.current_mask_index]['mask'] = {}
            self.mask_names[self.current_mask_index]['mask'][self.slice_index] = self.current_masks[
                self.current_display_mask_index]
            self.clear_segment()
            self.update_image(ax, canvas)
        elif event.keysym == 'Left':
            self.current_display_mask_index = (self.current_display_mask_index - 1) % len(self.current_masks)
            self.update_image(ax, canvas)
        elif event.keysym == 'Right':
            self.current_display_mask_index = (self.current_display_mask_index + 1) % len(self.current_masks)
            self.update_image(ax, canvas)
        elif event.keysym == 'Up':
            event.delta = 1
            self.on_scroll(event, ax, canvas)
        elif event.keysym == 'Down':
            event.delta = -1
            self.on_scroll(event, ax, canvas)


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

# TODO next
# 1. change so that 1-9 sets current label mode to that digit, then enter to save current mask
# 2. Scroll to next slice then use logits from previous slice's mask as input to next slice
# 3. Repeat for all slices. Need to get nearest mask logits for each slice
