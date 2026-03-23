import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def detect_dots_adaptive(image_path, min_area=100, max_area=50000, block_size=101, C=5, blur_size=5):
    """
    Detect circular dots using adaptive thresholding and contour analysis.
    This handles varying illumination better than global settings.
    Returns a list of tuples: (x, y, radius)
    """
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read the image: {image_path}")

    # Blur to reduce high-frequency noise
    img_blurred = cv.medianBlur(img, blur_size)

    # 1. Background Equalization / Adaptive Thresholding
    thresh = cv.adaptiveThreshold(
        img_blurred, 
        255, 
        cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv.THRESH_BINARY_INV, 
        block_size,  
        C
    )

    # 2. Distance Transform
    # To separate touching circles, we use the distance transform, which calculates
    # the distance from every white pixel to the nearest black pixel (background).
    # The center of a circle will have the highest distance (the peak).
    dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 5)

    # 3. Find Local Maxima (Centers of the dots)
    # By dilating the distance transform and finding where it equals the original,
    # we can pinpoint the exact localized peaks. This effortlessly splits touching circles 
    # because they will have two distinct peaks separated by a "valley" bridge.
    dilated_dist = cv.dilate(dist_transform, np.ones((7,7), np.uint8))
    
    min_radius = np.sqrt(min_area / np.pi)
    max_radius = np.sqrt(max_area / np.pi)
    
    # A peak is valid if it's a local maximum and falls within our radius thresholds
    centers_mask = (dist_transform == dilated_dist) & (dist_transform >= min_radius) & (dist_transform <= max_radius)
    
    # 4. Extract Coordinates & Apply Non-Maximum Suppression (NMS)
    # A single large, irregular dot might create multiple small peaks at its center.
    # We extract all peaks and then remove any smaller peak that falls inside the radius 
    # of a larger peak.
    centers_8u = np.uint8(centers_mask) * 255
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(centers_8u)
    
    raw_peaks = []
    for i in range(1, num_labels):
        x, y = int(centroids[i][0]), int(centroids[i][1])
        radius = int(dist_transform[y, x])
        raw_peaks.append((x, y, radius))
        
    # Sort peaks by radius in descending order
    raw_peaks.sort(key=lambda p: p[2], reverse=True)
    
    coordinates = []
    for current_peak in raw_peaks:
        cx, cy, cr = current_peak
        is_fragment = False
        
        # Check if this peak is inside the body of an already confirmed larger dot
        for confirmed_peak in coordinates:
            fx, fy, fr = confirmed_peak
            dist = np.sqrt((cx - fx)**2 + (cy - fy)**2)
            
            # If the distance between centers is less than the radius of the larger dot, 
            # it means this peak is just a fragmentation of the same dot.
            if dist <= fr:
                is_fragment = True
                break
                
        if not is_fragment:
            coordinates.append(current_peak)
        
    return coordinates

def map_dots_to_frame(coords, frame_path):
    expected_points = []
    with open(frame_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if not lines:
            return [(c[0], c[1], c[2], "Unknown", "Unknown") for c in coords]
            
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                col = int(parts[1])
                row = int(parts[2])
                pt_id = parts[3]
                name = parts[4].strip('"')
                expected_points.append({
                    'row': row, 'col': col, 'id': pt_id, 'name': name
                })
                
    if not expected_points:
        return [(c[0], c[1], c[2], "Unknown", "Unknown") for c in coords]
        
    num_expected = len(expected_points)
    max_row = max(p['row'] for p in expected_points)
    max_col = max(p['col'] for p in expected_points)
    
    top_dots = coords[:num_expected]
    extra_dots = coords[num_expected:]
    
    # Group into rows by Y
    top_dots_sorted_y = sorted(top_dots, key=lambda p: p[1])
    mapped_results = []
    
    for r_idx in range(max_row):
        row_slice = top_dots_sorted_y[r_idx * max_col : (r_idx + 1) * max_col]
        # Sort row by X
        row_slice_sorted_x = sorted(row_slice, key=lambda p: p[0])
        
        for c_idx, dot in enumerate(row_slice_sorted_x):
            mapped_pt = next((p for p in expected_points if p['row'] == r_idx + 1 and p['col'] == c_idx + 1), None)
            if mapped_pt:
                mapped_results.append((dot[0], dot[1], dot[2], mapped_pt['id'], mapped_pt['name']))
            else:
                mapped_results.append((dot[0], dot[1], dot[2], "Unknown", "Unknown"))
                
    # Append any extra, unmapped dots
    for dot in extra_dots:
        mapped_results.append((dot[0], dot[1], dot[2], "Unknown", "Unknown"))
        
    return mapped_results

if __name__ == "__main__":
    import csv
    
    image_file = 'test.tiff'
    output_csv = 'detected_dots.csv'
    
    print(f"Processing image: {image_file}")
    
    try:
        coords = detect_dots_adaptive(image_file, min_area=500, max_area=12000, block_size=151, C=10)
        print(f"Detected {len(coords)} dots.")
        
        # Save the extracted coordinates and information to a CSV file
        print(f"Saving coordinates to {output_csv}...")
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            # Write Header
            writer.writerow(['Dot ID', 'X Coordinate', 'Y Coordinate', 'Radius'])
            
            for idx, (x, y, r) in enumerate(coords):
                writer.writerow([idx + 1, x, y, r])
                print(f"Dot {idx + 1}: x={x}, y={y}, radius={r}")
                
        print(f"\nSuccessfully stored all locations to '{output_csv}'! You can now load this file into any other script, program, or Excel.")
            
        # Visualize the results
        original_img = cv.imread(image_file)
        if original_img is not None:
            if len(coords) > 0:
                for idx, (x, y, r) in enumerate(coords):
                    # Draw the outer circle
                    cv.circle(original_img, (x, y), r, (0, 255, 0), 2)
                    # Draw the center of the circle
                    cv.circle(original_img, (x, y), 2, (0, 0, 255), 3)
                    
            plt.figure(figsize=(10, 8))
            plt.imshow(cv.cvtColor(original_img, cv.COLOR_BGR2RGB))
            plt.title(f'Detected {len(coords)} Circular Dots')
            plt.axis('off')
            plt.show()
            
    except Exception as e:
        print(f"Error: {e}")