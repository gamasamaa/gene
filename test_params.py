import cv2 as cv
import numpy as np
from gene import detect_dots_adaptive

img_path = 'tests/test4.tiff'

print("Original script with default GUI params:")
try:
    coords = detect_dots_adaptive(img_path, min_area=500, max_area=12000, block_size=151, C=10)
    print(f"Found {len(coords)} dots")
except Exception as e:
    print("Error:", e)

# Test with modified blur
def custom_detect(image_path, min_area=500, max_area=12000, block_size=151, C=10, blur=5):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    img_blurred = cv.medianBlur(img, blur)
    thresh = cv.adaptiveThreshold(img_blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block_size, C)
    dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 5)
    dilated_dist = cv.dilate(dist_transform, np.ones((7,7), np.uint8))
    min_radius = np.sqrt(min_area / np.pi)
    max_radius = np.sqrt(max_area / np.pi)
    centers_mask = (dist_transform == dilated_dist) & (dist_transform >= min_radius) & (dist_transform <= max_radius)
    centers_8u = np.uint8(centers_mask) * 255
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(centers_8u)
    
    raw_peaks = []
    for i in range(1, num_labels):
        x, y = int(centroids[i][0]), int(centroids[i][1])
        radius = int(dist_transform[y, x])
        raw_peaks.append((x, y, radius))
    raw_peaks.sort(key=lambda p: p[2], reverse=True)
    
    coordinates = []
    for current_peak in raw_peaks:
        cx, cy, cr = current_peak
        is_fragment = False
        for confirmed_peak in coordinates:
            fx, fy, fr = confirmed_peak
            if np.sqrt((cx - fx)**2 + (cy - fy)**2) <= fr:
                is_fragment = True
                break
        if not is_fragment:
            coordinates.append(current_peak)
    return coordinates

print("\nTesting Custom Blur sizes:")
for b in [5, 11, 21, 31, 41]:
    for C in [10, 15, 20]:
        for block in [151, 301, 501]:
            try:
                coords = custom_detect(img_path, min_area=100, max_area=50000, block_size=block, C=C, blur=b)
                if 5 <= len(coords) <= 20:
                    print(f"Blur={b}, C={C}, Block={block} -> Found {len(coords)} dots")
            except Exception as e:
                pass
