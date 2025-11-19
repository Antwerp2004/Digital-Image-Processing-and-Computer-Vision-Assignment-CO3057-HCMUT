from os.path import isfile
import cv2
import json
import numpy as np
import logging
import glob
import os
import csv
from detect_board import find_corners, sort_corner_points

logger = logging.getLogger(__name__)


def evaluate(dataset_path, result_csv_path, experiment_name="baseline"):
    """
    Evaluates the corner detection algorithm on a given dataset and logs results to a CSV.
    Args:
        dataset_path (str): Path to the dataset folder (e.g., './Data/val/').
        result_csv_path (str): Path to the CSV file for logging results.
        experiment_name (str): A name to identify this experimental run (e.g., "canny_50_150").
    """
    total_boards = 0
    total_incorrect_corners = 0
    boards_with_no_mistakes = 0
    boards_with_le_one_mistake = 0
    total_pixel_error_sum = 0.0

    image_files = glob.glob(os.path.join(dataset_path, "*.png"))
    total_img = len(image_files)
    if total_img == 0:
        print(f"No images found at {dataset_path}. Please check the path.")
        return
    else:
        print("Finish generating images' paths.")

    for img_file in image_files:
        total_boards += 1
        img = cv2.imread(str(img_file))
        if img is None: continue
        base_name = os.path.basename(img_file)
        stem = os.path.splitext(base_name)[0]
        json_file = os.path.join(dataset_path, f"{stem}.json")
        
        with open(json_file, "r") as f:
            label = json.load(f)
        actual = np.array(label["corners"])

        incorrect_corners_on_this_board = 0
        mistake_pixel_threshold = (img.shape[0] + img.shape[1]) / 2 * 0.005

        try:
            predicted = find_corners(img)

        except Exception:
            predicted = None
        
        if predicted is not None:
            actual = sort_corner_points(actual)
            predicted = sort_corner_points(predicted)
            # print(actual - predicted)
            pixel_errors = np.linalg.norm(actual - predicted, axis=1)
            
            # Check each of the 4 corners for mistakes
            is_mistake_mask = pixel_errors > mistake_pixel_threshold
            incorrect_corners_on_this_board = np.sum(is_mistake_mask)
        else:
            # If find_corners failed, all 4 corners are considered incorrect
            incorrect_corners_on_this_board = 4
            # The maximum possible error is the length of the image diagonal.
            img_diagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
            pixel_errors = np.array([img_diagonal] * 4)

        # Update the aggregate counters
        total_pixel_error_sum += np.sum(pixel_errors)
        total_incorrect_corners += incorrect_corners_on_this_board
        if incorrect_corners_on_this_board == 0:
            boards_with_no_mistakes += 1
        if incorrect_corners_on_this_board <= 1:
            boards_with_le_one_mistake += 1

        if total_boards % 100 == 0:
          print(f'Total boards: {total_boards}, Total incorrect corners: {total_incorrect_corners}, Boards with no mistakes: {boards_with_no_mistakes}, Average pixel error: {total_pixel_error_sum / (total_boards * 4)}')

    # --- Calculate Final Statistics ---
    # Handle division by zero if no boards were processed
    if total_boards > 0:
        total_corners_evaluated = total_boards * 4

        # Metric 1: mean number of incorrect corners per board
        mean_incorrect_corners_per_board = total_incorrect_corners / total_boards
        
        # Metric 2: percentage of boards predicted with no mistakes
        percent_no_mistakes = (boards_with_no_mistakes / total_boards) * 100
        
        # Metric 3: percentage of boards predicted with <=1 mistakes
        percent_le_one_mistake = (boards_with_le_one_mistake / total_boards) * 100
        
        # Metric 4: per-corner error rate
        per_corner_error_rate = (total_incorrect_corners / total_corners_evaluated) * 100
        
        # Metric 7: average pixel error (per corner)
        avg_pixel_error = total_pixel_error_sum / total_corners_evaluated

    else:
        # Set all metrics to 0 if no data
        mean_incorrect_corners_per_board = 0
        percent_no_mistakes = 0
        percent_le_one_mistake = 0
        per_corner_error_rate = 0
        avg_pixel_error = 0

    # --- Print to Console ---
    print(f"\n--- Results for Experiment: {experiment_name} on {dataset_path} ---")
    print(f"Total boards evaluated: {total_boards}")
    print(f"1. Mean incorrect corners per board: {mean_incorrect_corners_per_board}")
    print(f"2. Boards with NO mistakes: {boards_with_no_mistakes} ({percent_no_mistakes}%)")
    print(f"3. Boards with <=1 mistake: {boards_with_le_one_mistake} ({percent_le_one_mistake}%)")
    print(f"4. Per-corner error rate: {per_corner_error_rate}%")
    print(f"5. Total boards with no mistakes (raw count): {boards_with_no_mistakes}")
    print(f"6. Total incorrect corners (raw count): {total_incorrect_corners}")
    print(f"7. Average pixel error per corner: {avg_pixel_error}")

    # --- Append to CSV ---
    header = [
        'experiment_name', 'dataset', 'total_boards',
        'mean_incorrect_corners_per_board', 'percent_no_mistakes', 'percent_le_one_mistake',
        'per_corner_error_rate', 'boards_with_no_mistakes', 'total_incorrect_corners',
        'avg_pixel_error'
    ]
    results = {
        'experiment_name': experiment_name,
        'dataset': dataset_path,
        'total_boards': total_boards,
        'mean_incorrect_corners_per_board': mean_incorrect_corners_per_board,
        'percent_no_mistakes': percent_no_mistakes,
        'percent_le_one_mistake': percent_le_one_mistake,
        'per_corner_error_rate': per_corner_error_rate,
        'boards_with_no_mistakes': boards_with_no_mistakes,
        'total_incorrect_corners': total_incorrect_corners,
        'avg_pixel_error': avg_pixel_error
    }

    existing_data = []
    experiment_found = False

    # Read existing data if the file exists
    if os.path.isfile(result_csv_path):
        with open(result_csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['experiment_name'] == experiment_name and row['dataset'] == dataset_path:
                    existing_data.append(results)
                    experiment_found = True
                    print(f"Overwriting results for experiment: '{experiment_name}'")
                else:
                    existing_data.append(row)
    if not experiment_found:
        existing_data.append(results)
        print(f"Adding new results for experiment: '{experiment_name}'")

    # Write the entire (potentially modified) data back to the file
    with open(result_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(existing_data)
    
    print(f"Results appended to {result_csv_path}")


if __name__ == "__main__":
    evaluate(dataset_path='./Data/train/', 
             result_csv_path='evaluation_log.csv',
             experiment_name='canny_80_350')