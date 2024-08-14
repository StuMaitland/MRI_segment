import os
import nibabel as nib
import numpy as np
import cv2
from tkinter import Tk, filedialog


def select_nii_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii *.nii.gz")])
    return file_path


def load_nii_file(file_path):
    nii_img = nib.load(file_path)
    return nii_img.get_fdata()


def create_output_folder(nii_file_path):
    base_dir = os.path.dirname(nii_file_path)
    output_folder = os.path.join(base_dir, "jpeg_slices")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def convert_slices_to_jpegs(nii_data, output_folder):
    num_slices = nii_data.shape[2]
    for i in range(num_slices):
        slice_data = nii_data[:, :, i]
        img_data = np.flipud(np.fliplr(slice_data))
        slice_data = np.rot90(img_data, k=-1, axes=(0, 1))
        slice_data_normalized = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX)
        slice_data_uint8 = slice_data_normalized.astype(np.uint8)
        output_path = os.path.join(output_folder, f"{i:03d}.jpg")
        cv2.imwrite(output_path, slice_data_uint8)


def convert_nii_to_jpegs(nii_file_path):
    nii_data = load_nii_file(nii_file_path)
    output_folder = create_output_folder(nii_file_path)
    convert_slices_to_jpegs(nii_data, output_folder)


def main():
    nii_file_path = select_nii_file()
    if not nii_file_path:
        print("No file selected.")
        return

    convert_slices_to_jpegs(nii_file_path)


if __name__ == "__main__":
    main()
