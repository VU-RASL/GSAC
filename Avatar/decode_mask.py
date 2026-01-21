import os
import numpy as np
import cv2

# Define input and output folders
input_folder = 'data/Customdata/rendong_1080_1080/render/seg/sapiens_1b/'
output_folder = 'data/Customdata/rendong_1080_1080/render/mask/'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process each .npy file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.npy'):
        # Load npy file (mask should be a binary or segmented array)
        file_path = os.path.join(input_folder, filename)
        mask = np.load(file_path)  # Directly loading .npy file

        # Convert to binary (255 for foreground, 0 for background)
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        binary_mask[mask > 0] = 255  # Foreground is white

        # Save the mask image as a **single-channel grayscale PNG**
        output_path = os.path.join(output_folder, filename.replace('.npy', '.png'))
        cv2.imwrite(output_path, binary_mask)  # No need to stack channels

print("Mask conversion completed!")
