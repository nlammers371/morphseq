import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to your mask file
mask_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/jpg_masks/20240404/masks/20240404_E05_0011_masks_emnum_1.jpg"

# Load mask as grayscale
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Print unique pixel values (labels)
unique_labels = np.unique(mask)
print("Unique labels in mask:", unique_labels)

# Create a color map for visualization
# 0 = background (black), 1 = embryo 1 (red), 2 = embryo 2 (green), etc.
colors = np.array([
    [0, 0, 0],      # background
    [255, 0, 0],    # embryo 1 (red)
    [0, 255, 0],    # embryo 2 (green)
    [0, 0, 255],    # embryo 3 (blue)
    [255, 255, 0],  # embryo 4 (yellow)
    [255, 0, 255],  # embryo 5 (magenta)
    [0, 255, 255],  # embryo 6 (cyan)
    [255, 255, 255] # embryo 7 (white)
], dtype=np.uint8)

# Map mask labels to colors
color_mask = colors[mask]

# Show the mask
plt.figure(figsize=(8, 8))
plt.title("Embryo Mask Visualization")
plt.imshow(color_mask)
plt.axis('off')
plt.show()
