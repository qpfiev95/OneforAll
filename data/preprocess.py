import os
import numpy as np
from PIL import Image

# Define the root folder where the data is stored
data_root = '@@ v Rx0001 (+10 dB) 000.mat'

# Accumulate the sum of pixel values for each channel
channel_sum = np.zeros(3)
channel_squared_sum = np.zeros(3)
num_pixels = 0

class_folders = sorted(os.listdir(data_root))
for class_folder in class_folders:
    class_path = os.path.join(data_root, class_folder)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path).convert("RGB")
            pixels = np.array(image).astype(np.float32) / 255.0
            channel_sum += np.sum(pixels, axis=(0, 1))
            channel_squared_sum += np.sum(pixels ** 2, axis=(0, 1))
            num_pixels += image.size[0] * image.size[1]

# Compute the mean and standard deviation
mean = channel_sum / num_pixels
std = np.sqrt((channel_squared_sum / num_pixels) - mean ** 2)

print("Mean:", mean)
print("Standard Deviation:", std)