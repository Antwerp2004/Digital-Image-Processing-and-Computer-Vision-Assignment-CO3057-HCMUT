import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from detect_board import find_corners, _detect_lines, _cluster_lines


def draw_hough_lines(img, lines, color=(0, 0, 255), thickness=2):
    """Helper function to draw lines from (rho, theta) format onto an image."""
    if lines is None:
        return
    img_out = img.copy()
    h, w = img.shape[:2]
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Calculate two points on the line to draw it across the entire image
        pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
        pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))
        cv2.line(img_out, pt1, pt2, color, thickness)
    return img_out


def run_and_visualize_stages(image_path, output_dir='visualizations/'):
    """
    Runs the detection pipeline on a single image and saves the intermediate
    visualizations for each stage.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load Image ---
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Image not found at {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_dir, '0_original.jpg'), img_bgr)

    # --- Stage 1: Preprocessing ---
    h, w, _ = img_bgr.shape
    target_width = 1200
    img_scale = target_width / w
    dims = (target_width, int(h * img_scale))
    resized_img = cv2.resize(img_bgr, dims)
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, '1_preprocess.jpg'), gray)

    # --- Stage 2: Edge and Line Detection ---
    # Canny Edge Detection
    edges = cv2.Canny(gray, 90, 400)
    cv2.imwrite(os.path.join(output_dir, '2.1_canny_edges.jpg'), edges)
    
    # Hough Transform
    all_lines = _detect_lines(edges)
    img_with_all_lines = draw_hough_lines(resized_img, all_lines, color=(0, 0, 255))
    cv2.imwrite(os.path.join(output_dir, '2.2_hough_lines_all.jpg'), img_with_all_lines)

    # --- Stage 3: Line Clustering and Filtering ---
    horizontal_lines, vertical_lines = _cluster_lines(all_lines)
    
    # Draw clustered lines (horizontal in blue, vertical in green)
    img_clustered_lines = draw_hough_lines(resized_img, horizontal_lines, color=(255, 0, 0)) # Blue
    img_clustered_lines = draw_hough_lines(img_clustered_lines, vertical_lines, color=(0, 255, 0)) # Green
    cv2.imwrite(os.path.join(output_dir, '3_clustered_lines.jpg'), img_clustered_lines)
    
    # Final step: Draw the detected corners on the original image
    try:
        # Call the modified function with visualize=True
        final_corners, warped_viz, warped_refined_viz = find_corners(cv2.imread(image_path), visualize=True)

        # STAGE 4 VISUALIZATION: Bird's-eye view with RANSAC inlier grid
        cv2.imwrite(os.path.join(output_dir, '4_warped_grid.jpg'), warped_viz)

        # STAGE 5 VISUALIZATION: Bird's-eye view with refined boundaries
        cv2.imwrite(os.path.join(output_dir, '5_refined_boundaries.jpg'), warped_refined_viz)

        img_final_bgr = cv2.imread(image_path)
        for x, y in final_corners:
            cv2.circle(img_final_bgr, (int(x), int(y)), radius=15, color=(0, 0, 255), thickness=-1)
        cv2.imwrite(os.path.join(output_dir, '6_final_corners.jpg'), img_final_bgr)
    except Exception as e:
        print(f"Could not find final corners: {e}")

    print(f"Visualization images saved to '{output_dir}' directory.")


# --- Run the Visualization ---
# IMPORTANT: Choose a good example image from your dataset
IMAGE_TO_VISUALIZE = './Data/test/0046.png' 
run_and_visualize_stages(IMAGE_TO_VISUALIZE)