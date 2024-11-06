import numpy as np
from skimage.measure import LineModelND, ransac
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import random

# Define the function to find multiple lines using RANSAC
def find_multiple_lines_ransac(data_array, max_lines=10, residual_threshold=5.0, n_iterations=5000):

    min_sam = 10

    labels = np.full(data_array.shape[0], 20)  # Initialize labels
    current_line_label = 1  # Start labeling lines from 1

    # Use a boolean mask to keep track of which points are still available
    available_points_mask = np.ones(data_array.shape[0], dtype=bool)

    fitted_models = {}

    # Loop to find multiple lines
    for _ in range(max_lines):
        # Filter points based on the available mask
        points = data_array[available_points_mask, :3]  # Use x, y, z coordinates

        if points.shape[0] > min_sam:
            min_samples = int(min(min_sam, points.shape[0]))
            # print(min_samples)
            model_class = LineModelND  # Reference the model class
            np.random.seed(42)
            random.seed(42)
            model, inlier_mask = ransac(points, model_class, min_samples=min_samples,
                                        residual_threshold=residual_threshold, max_trials=n_iterations)

            if np.sum(inlier_mask) < min_sam:
                print("Not enough inliers found, skipping to the next line.")
                continue

        else:
            break

        # If no inliers are found, break the loop
        if not np.any(inlier_mask):
            break

        inlier_indices = np.where(available_points_mask)[0][inlier_mask]

        # Assign current line label to inliers
        labels[inlier_indices] = current_line_label
        fitted_models[current_line_label] = model
        current_line_label += 1

        # Update the available points mask to exclude the inliers
        available_points_mask[inlier_indices] = False

        # If no more points left, break
        if not np.any(available_points_mask):  # Check if all points are exhausted
            break

    return labels, fitted_models

