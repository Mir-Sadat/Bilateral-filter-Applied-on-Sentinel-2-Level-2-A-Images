import os
import numpy as np
import rasterio
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# === Local Desktop Paths ===
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
tif_folder = os.path.join(desktop, "images")
output_folder = os.path.join(desktop, "output")
os.makedirs(output_folder, exist_ok=True)

# === Bilateral Filter Parameters ===
diameter = 9
sigma_color = 75
sigma_space = 75
mssim_scores = []

# === Load all TIFF files ===
tif_files = sorted([f for f in os.listdir(tif_folder) if f.lower().endswith(".tif")])

# === For Visualization of first 10 images ===
plt.figure(figsize=(12, min(10, len(tif_files)) * 4))

for idx, filename in enumerate(tif_files):
    filepath = os.path.join(tif_folder, filename)

    with rasterio.open(filepath) as src:
        bands = src.read()  # shape: (bands, height, width)
        profile = src.profile

    filtered_bands = np.zeros_like(bands)
    ssim_values = []

    for i in range(bands.shape[0]):
        original = bands[i].astype(np.float32)

        # Normalize for OpenCV
        min_val = original.min()
        max_val = original.max()
        if max_val - min_val == 0:
            original_norm = original_uint8 = np.zeros_like(original, dtype=np.uint8)
        else:
            original_norm = (original - min_val) / (max_val - min_val)
            original_uint8 = (original_norm * 255).astype(np.uint8)

        # Apply bilateral filter
        filtered_uint8 = cv2.bilateralFilter(original_uint8, diameter, sigma_color, sigma_space)

        # Restore original range
        filtered = filtered_uint8.astype(np.float32) / 255.0 * (max_val - min_val) + min_val
        filtered_bands[i] = filtered

        # SSIM calculation only for first 10 images
        if idx < 10:
            filtered_norm = (filtered - min_val) / (max_val - min_val) if max_val - min_val != 0 else filtered
            ssim_val = ssim(original_norm, filtered_norm, data_range=1.0)
            ssim_values.append(ssim_val)

    if idx < 10:
        mean_ssim = np.mean(ssim_values)
        mssim_scores.append((filename, mean_ssim))

    # Save filtered TIFF with original filename
    output_path = os.path.join(output_folder, filename)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(filtered_bands)

    # Visualization only for first 10 images
    if idx < 10:
        def normalize_image(band_array):
            band_array = band_array.astype(np.float32)
            min_val, max_val = np.percentile(band_array, (1, 99))
            return np.clip((band_array - min_val) / (max_val - min_val + 1e-5), 0, 1)

        if bands.shape[0] >= 3:
            orig_rgb = np.stack([
                normalize_image(bands[0]),
                normalize_image(bands[1]),
                normalize_image(bands[2])
            ], axis=-1)

            filt_rgb = np.stack([
                normalize_image(filtered_bands[0]),
                normalize_image(filtered_bands[1]),
                normalize_image(filtered_bands[2])
            ], axis=-1)

            # Display original
            plt.subplot(10, 2, 2 * idx + 1)
            plt.imshow(orig_rgb)
            plt.title(f"Original RGB - {filename}")
            plt.axis('off')

            # Display filtered
            plt.subplot(10, 2, 2 * idx + 2)
            plt.imshow(filt_rgb)
            plt.title(f"Bilateral Filtered (MSSIM={mean_ssim:.4f})")
            plt.axis('off')

# Show side-by-side images for first 10 files
if len(tif_files) > 0:
    plt.tight_layout()
    plt.show()

# === MSSIM Summary for first 10 images ===
print("\n=== MSSIM Summary for first 10 images ===")
for fname, score in mssim_scores:
    print(f"{fname}: {score:.4f}")

if mssim_scores:
    avg_mssim = np.mean([s[1] for s in mssim_scores])
    print(f"\nAverage MSSIM for first 10 images: {avg_mssim:.4f}")
else:
    print("No TIFF images found to process.")
