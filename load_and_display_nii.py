import tkinter as tk
from tkinter import filedialog
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

import segment
from segment import segment_image, show_mask


def select_nii_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")]
    )
    return file_path


def load_nii_file(file_path):
    # Load the .nii file
    img = nib.load(file_path)
    # Get the image data as a numpy array
    img_data = img.get_fdata()
    # img_data = np.flipud(np.fliplr(img_data))
    # img_data = np.rot90(img_data, k=-1, axes=(0, 1))
    return img_data


def update_image(slice_index, img_data, ax, canvas):
    ax.clear()
    ax.imshow(img_data[:, :, slice_index], cmap='bone')
    ax.axis('off')
    canvas.draw()


def convert_to_rgb(grayscale_slice):
    # Normalize the grayscale values to the range [0, 1]
    normalized_slice = (grayscale_slice - np.min(grayscale_slice)) / (np.max(grayscale_slice) - np.min(grayscale_slice))

    # Stack the normalized grayscale values along the third dimension to create an RGB image
    rgb_image = np.stack((normalized_slice,) * 3, axis=-1)
    rgb_image = rgb_image.astype(np.float32)  # Convert to float32
    return rgb_image


def on_scroll(event, img_data, ax, canvas):
    if event.delta > 0:
        on_scroll.slice_index = min(on_scroll.slice_index + 1, img_data.shape[2] - 1)
    else:
        on_scroll.slice_index = max(on_scroll.slice_index - 1, 0)
    update_image(on_scroll.slice_index, img_data, ax, canvas)


def on_click(event, img_data, ax, canvas):
    x, y = int(event.xdata), int(event.ydata)
    slice_index = on_scroll.slice_index
    rgb_image = convert_to_rgb(img_data[:, :, slice_index])
    mask = segment_image(rgb_image, [[x, y]], [1])
    segment.show_mask(mask, ax)
    canvas.draw()


# Example usage
if __name__ == "__main__":
    file_path = select_nii_file()
    if file_path:
        img_data = load_nii_file(file_path)

        root = tk.Tk()
        fig, ax = plt.subplots()
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        on_scroll.slice_index = img_data.shape[2] // 2  # Start with the middle slice
        update_image(on_scroll.slice_index, img_data, ax, canvas)

        root.bind("<MouseWheel>", lambda event: on_scroll(event, img_data, ax, canvas))
        canvas.mpl_connect("button_press_event", lambda event: on_click(event, img_data, ax, canvas))
        root.mainloop()
