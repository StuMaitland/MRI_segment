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
        ax.axis('off')
        canvas.draw()

    def convert_to_rgb(self, grayscale_slice):
        # Normalize the grayscale values to the range [0, 1]
        normalized_slice = (grayscale_slice - np.min(grayscale_slice)) / (
                np.max(grayscale_slice) - np.min(grayscale_slice))

        # Stack the normalized grayscale values along the third dimension to create an RGB image
        rgb_image = np.stack((normalized_slice,) * 3, axis=-1)
        rgb_image = rgb_image.astype(np.float32)  # Convert to float32
        return rgb_image

    def on_scroll(self, event, ax, canvas):
        if event.delta > 0:
            self.slice_index = min(self.slice_index + 1, self.img_data.shape[2] - 1)
        else:
            self.slice_index = max(self.slice_index - 1, 0)
        self.update_image(ax, canvas)

    def on_click(self, event, ax, canvas):
        print(f' button: {event.button}')
        x, y = int(event.xdata), int(event.ydata)
        rgb_image = self.convert_to_rgb(self.img_data[:, :, self.slice_index])
        mask = segment_image(self.predictor, rgb_image, [[x, y]], [1])
        segment.show_mask(mask, ax)
        canvas.draw()


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
    print(f"Event connection ID: {cid}")
    root.mainloop()

# Pseudocode for next steps:
# Left click- add point to array, start segmentation
# Right click- add a not-point, start segmentation
# Press delete key- remove last point
# Press number key- save most recent mask as that number
# Press enter- save output of all masks
