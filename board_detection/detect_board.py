from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import cv2
import numpy as np
import typing

def find_corners(img: np.ndarray, visualize=False) -> np.ndarray:
    """
    Main function to orchestrate the entire board detection pipeline.
    Takes an image and returns the four outer corner points of the chessboard.
    If visualize=True, returns intermediate results for plotting.
    """
    # --- STAGE 1: PREPROCESSING ---
    # Step 1.1: Normalize image width to 1200px.
    # This ensures that all subsequent fixed-pixel parameters (like thresholds) behave consistently regardless of the original image's resolution.
    # TARGET_WIDTH = 1200px is finetuned. (Values: 800, 1000, 1200, 1500)
    TARGET_WIDTH = 1200
    h, w, _ = img.shape
    img_scale = TARGET_WIDTH / w
    dims = (TARGET_WIDTH, int(h * img_scale))
    img = cv2.resize(img, dims)

    # Keep a color copy of the resized image for visualization
    resized_color_img = cv2.resize(img.copy(), dims)
    
    # Step 1.2: Convert to grayscale for edge detection.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # --- STAGE 2: EDGE & LINE DETECTION ---
    # Step 2.1: Detect all prominent edges using the Canny algorithm.
    # This creates a binary image showing pixels with sharp intensity changes.
    # Canny thresholds are finetuned (Values: ((50, 150), (90, 400), (100, 200), (120, 300), (120, 350), (90, 350), (90, 400), (120, 400)).
    edges = cv2.Canny(gray, 80, 350)
    
    # Step 2.2: Use the Hough Transform on the edge map to find all potential straight lines.
    lines = _detect_lines(edges)
    if lines.shape[0] > 400:
        raise Exception("too many lines in the image")


    # --- STAGE 3: LINE CLUSTERING & FILTERING ---
    # Step 3.1: Cluster the detected lines into two main groups: horizontal and vertical.
    all_horizontal_lines, all_vertical_lines = _cluster_lines(lines)

    # Step 3.2: Filter out redundant/noisy lines to get ~9 clean grid lines for each direction.
    horizontal_lines = _eliminate_similar_lines(
        all_horizontal_lines, all_vertical_lines)
    vertical_lines = _eliminate_similar_lines(
        all_vertical_lines, all_horizontal_lines)


    # --- STAGE 4: GRID RECONSTRUCTION (RANSAC) ---
    # Step 4.1: Calculate all intersection points between the filtered lines to form a distorted grid.
    all_intersection_points = _get_intersection_points(horizontal_lines, vertical_lines)

    best_num_inliers = 0
    best_configuration = None
    iterations = 0
    # Step 4.2: Use RANSAC to find the best perspective transformation (homography) that
    # maps the distorted grid to a perfect square grid. This is a "guess and check" loop.
    while iterations < 200 or best_num_inliers < 30:
        # 4.2a: Randomly sample a quadrilateral from the grid intersections.
        row1, row2 = _choose_from_range(len(horizontal_lines))
        col1, col2 = _choose_from_range(len(vertical_lines))
        # 4.2b: Calculate the transform that would make this random quad a perfect square ("bird's-eye view").
        transformation_matrix = _compute_homography(all_intersection_points,
                                                    row1, row2, col1, col2)
        # 4.2c: Apply this "guess" transform to ALL intersection points.
        warped_points = _warp_points(
            transformation_matrix, all_intersection_points)
        # 4.2d: Count how many points ('inliers') fit a perfect grid in this warped view.
        warped_points, intersection_points, horizontal_scale, vertical_scale = _discard_outliers(warped_points, all_intersection_points)
        num_inliers = np.prod(warped_points.shape[:-1])
        
        # 4.2e: If this transform produced the most inliers so far, save it as the best one.
        if num_inliers > best_num_inliers:
            warped_points *= np.array((horizontal_scale, vertical_scale))
            configuration = _quantize_points(warped_points, intersection_points)
            # Unpack configuration and recalculate inliers after quantization
            (xmin, xmax, ymin, ymax), scale, quantized_points, intersection_points, warped_img_size = configuration
            num_inliers = np.prod(quantized_points.shape[:-1])
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_configuration = configuration
        iterations += 1
        if iterations > 10000:
            raise Exception("RANSAC produced no viable results")


    # --- STAGE 5: BOUNDARY REFINEMENT ---
    # Step 5.1: Retrieve the best grid configuration found by RANSAC.
    (xmin, xmax, ymin, ymax), scale, quantized_points, intersection_points, warped_img_size = best_configuration

    # Step 5.2: Recompute the final, most accurate transformation matrix using ALL the inlier points.
    transformation_matrix = compute_transformation_matrix(intersection_points, quantized_points)
    
    # Step 5.3: Create the clean "bird's-eye" view of the board using the best transform.
    dims = tuple(warped_img_size.astype(np.int32))
    warped_color = cv2.warpPerspective(resized_color_img, transformation_matrix, dims)
    warped = cv2.warpPerspective(gray, transformation_matrix, dims)
    
    # Create the visualization for Stage 4: Bird's-eye view with RANSAC inlier grid.
    warped_viz_stage4 = warped_color.copy()
    for point in quantized_points.reshape(-1, 2):
        px, py = point
        cv2.circle(warped_viz_stage4, (int(px), int(py)), radius=10, color=(0, 255, 0), thickness=3)

    borders = np.zeros_like(gray); borders[3:-3, 3:-3] = 1
    warped_borders = cv2.warpPerspective(borders, transformation_matrix, dims)
    warped_mask = warped_borders == 1

    # Step 5.4: Refine the exact outer boundaries by finding the strongest edges in the warped image.
    xmin, xmax = _compute_vertical_borders(warped, warped_mask, scale, xmin, xmax)
    scaled_xmin, scaled_xmax = (int(x * scale[0]) for x in (xmin, xmax))
    warped_mask[:, :scaled_xmin] = warped_mask[:, scaled_xmax:] = False
    ymin, ymax = _compute_horizontal_borders(warped, warped_mask, scale, ymin, ymax)

    # Create the visualization for Stage 5: Bird's-eye view with refined boundaries.
    warped_viz_stage5 = warped_color.copy()
    translation = np.array([-np.min(quantized_points.reshape(-1, 2)[:,0])/scale[0] + 5, 
                            -np.min(quantized_points.reshape(-1, 2)[:,1])/scale[1] + 5])
    corners_warped = np.array([[xmin, ymin], [xmax, ymin], 
                               [xmax, ymax], [xmin, ymax]]).astype(np.float32)
    corners_warped = (corners_warped - translation) * scale
    cv2.polylines(warped_viz_stage5, [corners_warped.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=5)


    # --- STAGE 6: FINAL CORNER PROJECTION ---
    # Step 6.1: Define the final corners in the clean, warped coordinate space.
    corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]).astype(np.float32)
    corners = corners * scale
    
    # Step 6.2: Project these refined corners back onto the original, distorted image using the inverse transformation.
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)
    img_corners = _warp_points(inverse_transformation_matrix, corners)
    
    # Step 6.3: Scale the corners back to the original image's dimensions.
    img_corners /= img_scale

    if visualize:
        return sort_corner_points(img_corners), warped_viz_stage4, warped_viz_stage5
    else:
        return sort_corner_points(img_corners)



def _detect_lines(edges: np.ndarray) -> np.ndarray:
    """Uses Hough Transform to find lines and filters them to keep only near-horizontal/vertical ones."""
    # Step 1: Apply Hough Transform. `threshold=150` means a line needs at least 150 edge pixel "votes".
    # `threshold=175` is finetuned (Values: 100, 125, 150, 165, 175, 180, 200).
    lines = cv2.HoughLines(edges, 1, np.pi/360, threshold=150)
    lines = lines.squeeze(axis=-2)

    # Step 2: Standardize line representation so rho is always positive.
    lines = _fix_negative_rho_in_hesse_normal_form(lines)

    # Step 3: Filter out lines that are not close to horizontal (90 deg) or vertical (0 deg).
    threshold = np.deg2rad(30)
    vmask = np.abs(lines[:, 1]) < threshold
    hmask = np.abs(lines[:, 1] - np.pi / 2) < threshold
    mask = vmask | hmask
    lines = lines[mask]
    return lines



def _fix_negative_rho_in_hesse_normal_form(lines: np.ndarray) -> np.ndarray:
    """A line (rho, theta) is the same as (-rho, theta-pi).
    This function enforces the positive rho convention."""
    lines = lines.copy()
    neg_rho_mask = lines[..., 0] < 0
    lines[neg_rho_mask, 0] = - lines[neg_rho_mask, 0]
    lines[neg_rho_mask, 1] = lines[neg_rho_mask, 1] - np.pi
    return lines



def _absolute_angle_difference(x, y):
    """Calculates the smallest angle between two lines, used as a distance metric for clustering."""
    diff = np.mod(np.abs(x - y), 2*np.pi)
    return np.min(np.stack([diff, np.pi - diff], axis=-1), axis=-1)



def _sort_lines(lines: np.ndarray) -> np.ndarray:
    """Sorts lines by their distance from the origin (rho) for stable clustering."""
    if lines.ndim == 0 or lines.shape[-2] == 0:
        return lines
    rhos = lines[..., 0]
    sorted_indices = np.argsort(rhos)
    return lines[sorted_indices]



def _cluster_lines(lines: np.ndarray):
    """Separates a list of lines into two clusters: horizontal and vertical, based on their angles."""
    lines = _sort_lines(lines)
    # Step 1: Calculate the angle difference between every pair of lines.
    thetas = lines[..., 1].reshape(-1, 1)
    distance_matrix = pairwise_distances(thetas, thetas, metric=_absolute_angle_difference)

    # Step 2: Use Agglomerative Clustering to group all lines into exactly two clusters based on their angle.
    agg = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average")
    clusters = agg.fit_predict(distance_matrix)

    # Step 3: Determine which cluster is "horizontal" and which is "vertical".
    # Vertical lines have an angle closer to 0 or pi. Horizontal are closer to pi/2.
    angle_with_y_axis = _absolute_angle_difference(thetas, 0.)
    if angle_with_y_axis[clusters == 0].mean() > angle_with_y_axis[clusters == 1].mean():
        hcluster, vcluster = 0, 1
    else:
        hcluster, vcluster = 1, 0

    horizontal_lines = lines[clusters == hcluster]
    vertical_lines = lines[clusters == vcluster]
    return horizontal_lines, vertical_lines



def _eliminate_similar_lines(lines: np.ndarray, perpendicular_lines: np.ndarray) -> np.ndarray:
    """Groups nearby, parallel lines (which are redundant) and selects a single median line from each group."""
    # Step 1: Project the lines onto a common axis (the mean of the perpendicular lines) to get a 1D position.
    perp_rho, perp_theta = perpendicular_lines.mean(axis=0)
    rho, theta = np.moveaxis(lines, -1, 0)
    intersection_points = get_intersection_point(
        rho, theta, perp_rho, perp_theta)
    intersection_points = np.stack(intersection_points, axis=-1)

    # Step 2: Use DBSCAN to cluster lines that are physically close together in the image.
    # `eps=12` means lines whose intersections are within 12 pixels are grouped.
    # `eps=12` is finetuned (Values: 8, 10, 12, 15, 20).
    clustering = DBSCAN(eps=12, min_samples=1).fit(intersection_points)

    # Step 3: For each cluster of redundant lines, find the one in the middle (the median) and keep only that one.
    filtered_lines = []
    for c in range(clustering.labels_.max() + 1):
        lines_in_cluster = lines[clustering.labels_ == c]
        rho = lines_in_cluster[..., 0]
        median = np.argsort(rho)[len(rho)//2]
        filtered_lines.append(lines_in_cluster[median])
    return np.stack(filtered_lines)



def get_intersection_point(rho1: np.ndarray, theta1: np.ndarray, rho2: np.ndarray, theta2: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Calculates the (x, y) intersection point of two lines given in (rho, theta) form (Hough space)."""
    # rho1 = x cos(theta1) + y sin(theta1)
    # rho2 = x cos(theta2) + y sin(theta2)
    cos_t1 = np.cos(theta1); cos_t2 = np.cos(theta2)
    sin_t1 = np.sin(theta1); sin_t2 = np.sin(theta2)
    # Using the determinant solution for a system of linear equations.
    x = (sin_t1 * rho2 - sin_t2 * rho1) / (cos_t2 * sin_t1 - cos_t1 * sin_t2)
    y = (cos_t1 * rho2 - cos_t2 * rho1) / (sin_t2 * cos_t1 - sin_t1 * cos_t2)
    return x, y



def _choose_from_range(upper_bound: int, n: int = 2):
    """Utility for RANSAC: randomly selects 'n' unique indices from a range."""
    return np.sort(np.random.choice(np.arange(upper_bound), (n,), replace=False), axis=-1)



def _get_intersection_points(horizontal_lines: np.ndarray, vertical_lines: np.ndarray) -> np.ndarray:
    """Computes the intersection point for every horizontal line with every vertical line, creating a grid."""
    rho1, theta1 = np.moveaxis(horizontal_lines, -1, 0)
    rho2, theta2 = np.moveaxis(vertical_lines, -1, 0)
    # Create a grid of all rho/theta pairs to calculate all intersections at once (vectorized).
    rho1, rho2 = np.meshgrid(rho1, rho2, indexing="ij")
    theta1, theta2 = np.meshgrid(theta1, theta2, indexing="ij")
    intersection_points = get_intersection_point(rho1, theta1, rho2, theta2)
    intersection_points = np.stack(intersection_points, axis=-1)
    return intersection_points



def compute_transformation_matrix(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Compute the 3x3 perspective transformation matrix (homography) using OpenCV."""
    transformation_matrix, _ = cv2.findHomography(src_points.reshape(-1, 2),
                                                  dst_points.reshape(-1, 2))
    return transformation_matrix



def _compute_homography(intersection_points: np.ndarray, row1: int, row2: int, col1: int, col2: int):
    """Sets up the 4 source points (from image) and 4 destination points (a perfect square) for findHomography."""
    p1 = intersection_points[row1, col1]  # top left
    p2 = intersection_points[row1, col2]  # top right
    p3 = intersection_points[row2, col2]  # bottom right
    p4 = intersection_points[row2, col1]  # bottom left
    # Source points are the randomly sampled corners from the distorted grid.
    src_points = np.stack([p1, p2, p3, p4])
    # Destination points define a perfect unit square.
    dst_points = np.array([[0, 0],  # top left
                           [1, 0],  # top right
                           [1, 1],  # bottom right
                           [0, 1]])  # bottom left
    return compute_transformation_matrix(src_points, dst_points)



def _warp_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Applies a perspective transformation matrix to a set of 2D points."""
    points = to_homogenous_coordinates(points)
    warped_points = points @ transformation_matrix.T
    return from_homogenous_coordinates(warped_points)



def _find_best_scale(values: np.ndarray, scales: np.ndarray = np.arange(1, 9)):
    """
    Part of RANSAC. Guesses the grid size (1x1 to 8x8) by finding the scale that makes the most warped points align with an integer grid.
    """
    scales = np.sort(scales)
    # Step 1: Try scaling the warped coordinates by each possible grid size (1..8).
    scaled_values = np.expand_dims(values, axis=-1) * scales
    # Step 2: Calculate the distance of each point to the nearest integer. A small distance means it fits the grid well.
    diff = np.abs(np.rint(scaled_values) - scaled_values)

    # Step 3: A point is an "inlier" if it's very close to an integer after scaling.
    inlier_mask = diff < 0.1 / scales
    # Step 4: Count the inliers for each of the 8 possible scales.
    num_inliers = np.sum(inlier_mask, axis=tuple(range(inlier_mask.ndim - 1)))
    best_num_inliers = np.max(num_inliers)

    # Step 5: Robustly choose the smallest scale that's at least 85% as good as the best scale.
    index = np.argmax(num_inliers > 0.85 * best_num_inliers)
    return scales[index], inlier_mask[..., index]



def _discard_outliers(warped_points: np.ndarray, intersection_points: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float, float]:
    """Part of RANSAC: Finds the grid scale (e.g., 8x8 squares) and removes points that don't align with the grid."""
    # Step 1: Guess the number of squares horizontally and vertically by finding the best scale.
    horizontal_scale, horizontal_mask = _find_best_scale(warped_points[..., 0])
    vertical_scale, vertical_mask = _find_best_scale(warped_points[..., 1])
    mask = horizontal_mask & vertical_mask

    # Step 2: Filter out any rows or columns of points that don't have enough inliers (70%).
    num_rows_to_consider = np.any(mask, axis=-1).sum()
    num_cols_to_consider = np.any(mask, axis=-2).sum()
    rows_to_keep = mask.sum(axis=-1) / num_rows_to_consider > 0.7
    cols_to_keep = mask.sum(axis=-2) / num_cols_to_consider > 0.7

    warped_points = warped_points[rows_to_keep][:, cols_to_keep]
    intersection_points = intersection_points[rows_to_keep][:, cols_to_keep]
    return warped_points, intersection_points, horizontal_scale, vertical_scale



def _quantize_points(warped_scaled_points: np.ndarray, intersection_points: np.ndarray) -> typing.Tuple[tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Takes the set of inlier points, removes duplicates, and maps them to a clean integer grid."""
    # Step 1: Average the positions of the warped points to get clean line positions.
    mean_col_xs = warped_scaled_points[..., 0].mean(axis=0)
    mean_row_ys = warped_scaled_points[..., 1].mean(axis=1)
    col_xs = np.rint(mean_col_xs).astype(np.int32)
    row_ys = np.rint(mean_row_ys).astype(np.int32)

    # Step 2: Remove any duplicate rows/columns that may have resulted from the rounding.
    col_xs, col_indices = np.unique(col_xs, return_index=True)
    row_ys, row_indices = np.unique(row_ys, return_index=True)
    intersection_points = intersection_points[row_indices][:, col_indices]

    # Step 3: Find the min/max extents of the detected grid and ensure it's not larger than 8x8 squares.
    xmin, xmax, ymin, ymax = col_xs.min(), col_xs.max(), row_ys.min(), row_ys.max()

    # Ensure we a have a maximum of 9 rows/cols
    # This loop may over-trim (e.g., turn a 10-line grid into an 8-line grid), creating an undersized grid. This is INTENTIONAL.
    while xmax - xmin > 8: xmax -= 1; xmin += 1
    while ymax - ymin > 8: ymax -= 1; ymin += 1
    # The subsequent _compute_borders function will handle the "Fine Expand" step,
    # expanding any undersized grid back to the correct 8x8 size using image gradients.

    col_mask = (col_xs >= xmin) & (col_xs <= xmax)
    row_mask = (row_ys >= ymin) & (row_ys <= ymax)
    # Discard
    col_xs, row_ys = col_xs[col_mask], row_ys[row_mask]
    intersection_points = intersection_points[row_mask][:, col_mask]

    # Step 4: Create a final, perfect grid of destination points for the homography.
    quantized_points = np.stack(np.meshgrid(col_xs, row_ys), axis=-1)
    # Transform in warped space
    translation = -np.array([xmin, ymin]) + 5
    scale = np.array([50, 50])

    scaled_quantized_points = (quantized_points + translation) * scale
    xmin, ymin = np.array((xmin, ymin)) + translation
    xmax, ymax = np.array((xmax, ymax)) + translation
    warped_img_size = (np.array((xmax, ymax)) + 5) * scale
    return (xmin, xmax, ymin, ymax), scale, scaled_quantized_points, intersection_points, warped_img_size



def _compute_vertical_borders(warped: np.ndarray, mask: np.ndarray, scale: np.ndarray, xmin: int, xmax: int) -> typing.Tuple[int, int]:
    """Refines the left/right boundaries of the board by finding the strongest vertical edges in the warped image."""
    # Step 1: Calculate the vertical gradient ( Sobel X-derivative) of the warped image.
    G_x = np.abs(cv2.Sobel(warped, cv2.CV_64F, 1, 0, ksize=3))
    G_x[~mask] = 0
    G_x = G_x / G_x.max() * 255; G_x = G_x.astype(np.uint8)
    # Step 2: Apply Canny to the gradient image to get clean vertical edge lines.
    G_x = cv2.Canny(G_x, 100, 200); G_x[~mask] = 0

    def get_nonmax_supressed(x):
        x = (x * scale[0]).astype(np.int32)
        thresh = 2 # can also use 1?
        return G_x[:, x-thresh:x+thresh+1].max(axis=1)

    # Step 3: While we haven't found 8 squares (9 lines), check if the edge response is stronger
    # just outside the current left boundary or the right boundary, and expand in that direction.
    while xmax - xmin < 8:
        top = get_nonmax_supressed(xmax + 1)
        bottom = get_nonmax_supressed(xmin - 1)
        if top.sum() > bottom.sum(): xmax += 1
        else: xmin -= 1
    return xmin, xmax



def _compute_horizontal_borders(warped: np.ndarray, mask: np.ndarray, scale: np.ndarray, ymin: int, ymax: int) -> typing.Tuple[int, int]:
    """Refines the top/bottom boundaries of the board by finding the strongest horizontal edges in the warped image."""
    # Step 1: Calculate the horizontal gradient (Sobel Y-derivative) of the warped image.
    G_y = np.abs(cv2.Sobel(warped, cv2.CV_64F, 0, 1, ksize=3))
    G_y[~mask] = 0
    G_y = G_y / G_y.max() * 255; G_y = G_y.astype(np.uint8)
    # Step 2: Apply Canny to get clean horizontal edge lines.
    G_y = cv2.Canny(G_y, 120, 300); G_y[~mask] = 0

    def get_nonmax_supressed(y):
        y = (y * scale[1]).astype(np.int32)
        thresh = 2
        return G_y[y-thresh:y+thresh+1].max(axis=0)

    # Step 3: While we haven't found 8 squares, expand the top or bottom boundary based on edge strength.
    while ymax - ymin < 8:
        top = get_nonmax_supressed(ymax + 1)
        bottom = get_nonmax_supressed(ymin - 1)
        if top.sum() > bottom.sum(): ymax += 1
        else: ymin -= 1
    return ymin, ymax
    
    

def to_homogenous_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Converts 2D Cartesian coordinates (x, y) to 3D homogenous coordinates (x, y, 1) for matrix multiplication."""
    return np.concatenate([coordinates, np.ones((*coordinates.shape[:-1], 1))], axis=-1)



def from_homogenous_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Converts 3D homogenous coordinates back to 2D Cartesian by dividing by the z component."""
    return coordinates[..., :2] / coordinates[..., 2, np.newaxis]



def sort_corner_points(points: np.ndarray) -> np.ndarray:
    """Sorts the 4 corner points into a consistent order: top-left, top-right, bottom-right, bottom-left."""
    # Step 1: Sort primarily by y-coordinate. Top two points come first.
    points = points[points[:, 1].argsort()]
    # Step 2: For the top two points, sort by x-coordinate to find top-left and top-right.
    points[:2] = points[:2][points[:2, 0].argsort()]
    # Step 3: For the bottom two points, sort by x-coordinate in reverse to find bottom-right and bottom-left.
    points[2:] = points[2:][points[2:, 0].argsort()[::-1]]
    return points