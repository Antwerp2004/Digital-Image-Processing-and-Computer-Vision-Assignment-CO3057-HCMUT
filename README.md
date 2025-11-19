# Project: Chessboard Corner Detection using a Classical Computer Vision Pipeline

**Course:** Digital Image Processing & Computer Vision  
**Group:** L01_3  
**Members:** Đỗ Hoàng Quân, Nguyễn Quang Minh

---

## 1. Project Overview

This project implements and evaluates a classical, multi-stage computer vision pipeline to solve the problem of **chessboard localization**. The primary goal is to automatically and accurately detect the four outer corners of a physical chessboard from a single 2D image.

The implementation is based on the methodologies described in the academic papers by Wölflein & Arandjelović (2021) ([link to paper](https://www.mdpi.com/2313-433X/7/6/94)). The pipeline does not use any deep learning models for this task; instead, it relies on a sequence of traditional image processing algorithms, including Canny Edge Detection, Hough Transform, Agglomerative & DBSCAN Clustering, and the RANSAC algorithm for robust homography estimation.

The core of this project focuses on the **systematic tuning of the pipeline's hyperparameters** to optimize its performance, with results measured against a ground-truth dataset.

## 2. File Structure

The project is organized as follows. The main execution logic is contained within the `Execution.ipynb` notebook, which calls functions from the `.py` scripts.

```
.
├── board_detection/
│   ├── detect_board.py     # Core pipeline logic for corner detection
│   ├── evaluate.py         # Script for evaluating performance and logging results
│   └── visualize.py        # Script to generate intermediate visualization images
│
├── visualizations/         # Output directory for images showing pipeline stages
│   ├── 0_original.jpg
│   ├── 1_preprocess.jpg
│   └── ... (and other intermediate images)
│
├── Execution.ipynb         # Main Jupyter/Colab notebook for running experiments
├── evaluation_log.csv      # CSV file logging the results of all experiments
├── README.md               # This file
└── requirements.txt        # List of Python dependencies
```

## 3. Dataset

This project uses the **"Rendered Chess Game State Images"** dataset created by Wölflein & Arandjelović (2021). It contains 4,888 high-resolution rendered images of chess positions with variations in lighting and camera angles. Each image is paired with a `.json` file containing ground-truth annotations, including the coordinates of the four board corners.

### **Important: Getting the Data**

The dataset is large and is **not included** in this GitHub repository. To run the evaluation and visualization code, you must first download and set up the data.

1.  **Download the Dataset:** The dataset is available from the Open Science Framework (OSF) at the following link:
    *   [https://osf.io/xf3ka/](https://osf.io/xf3ka/)

2.  **Folder Structure:** After downloading and unzipping, please organize the data into a `Data` folder at the same level as the `Code` folder. The required structure is:
    ```
    /Computer Vision Assignment/
    ├── Code/         (This repository)
    └── Data/
        ├── train/    # Contains training images and .json files
        ├── val/      # Contains validation images and .json files
        └── test/     # Contains test images and .json files
    ```

## 4. How to Run the Project

This project is designed to be run in a Google Colab environment for ease of use and access to pre-configured Python environments.

### 4.1. Setup in Google Colab

1.  **Upload to Google Drive:** Upload the entire `Code` folder and the `Data` folder (from the previous step) to your Google Drive, maintaining the directory structure described above.
2.  **Open the Notebook:** In Google Drive, navigate to `Code/` and open the `Execution.ipynb` notebook in Google Colab.

### 4.2. Running the Experiments

The `Execution.ipynb` notebook is the main entry point for this project. It is structured with clear cells to:
1.  **Mount Google Drive** and navigate to the project directory.
2.  **Install Dependencies** from `requirements.txt`.
3.  **Run a Single Image Visualization:** Generate the intermediate images for one example to visually inspect the pipeline's stages.
4.  **Run Parameter Tuning Experiments:** Execute the main evaluation script (`evaluate.py`) with different parameter configurations. The results of these experiments are automatically logged to `evaluation_log.csv`.
5.  **Run Final Evaluation:** Perform the final model selection on the `val` set and the final performance test on the `test` set.

Follow the instructions and run the cells in the notebook in order.

## 5. Key Libraries

The project relies on the following major Python libraries. See `requirements.txt` for specific versions.
*   **OpenCV (`opencv-python`):** For all core image processing tasks.
*   **NumPy:** For numerical operations and data representation.
*   **Scikit-learn:** For the clustering algorithms (Agglomerative and DBSCAN).
*   **Matplotlib:** For visualizing results.
*   **gspread & google-auth:** (Used in the notebook) For logging results directly to a Google Sheet.