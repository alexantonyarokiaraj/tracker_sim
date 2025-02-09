import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from libraries import DataArray

def calculate_cluster_metrics(data_array, beam_zone_low, beam_zone_high):
    # Separate columns for easy access
    x, y, z = data_array[:, DataArray.X.value], data_array[:, DataArray.Y.value], data_array[:, DataArray.Z.value]
    charges, true_labels, ransac_labels, gmm_labels = data_array[:, DataArray.Q.value], data_array[:, DataArray.true_labels_sim.value], data_array[:, DataArray.ransac_labels.value], data_array[:, DataArray.gmm_labels.value]

    # Group points by GMM labels
    unique_gmm_labels = np.unique(gmm_labels)
    clusters = {label: data_array[gmm_labels == label] for label in unique_gmm_labels}

    # Separate clusters into 'beam' and 'track' based on y-values
    beam_clusters = {label: cluster for label, cluster in clusters.items() if is_beam(cluster, beam_zone_low, beam_zone_high)}
    track_clusters = {label: cluster for label, cluster in clusters.items() if not is_beam(cluster, beam_zone_low, beam_zone_high)}

    # Calculate the p-value metric for each pair of beam and track clusters
    beam_metrics = calculate_pair_p_values(beam_clusters, len(unique_gmm_labels))
    track_metrics = calculate_pair_p_values(track_clusters, len(unique_gmm_labels))
    beam_track_metrics = calculate_cross_p_values(beam_clusters, track_clusters, len(unique_gmm_labels))

    return beam_metrics, track_metrics, beam_track_metrics

# Helper function to determine if a cluster is a 'beam' based on y-values
def is_beam(cluster, beam_zone_low, beam_zone_high):
    y_values = cluster[:, 1]  # Assuming y is the second column
    mean_y = np.mean(y_values)
    return (beam_zone_low <= mean_y < beam_zone_high)

# Helper function to calculate p-values for each pair of clusters
def calculate_pair_p_values(clusters, unique_gmm_labels):
    labels = list(clusters.keys())
    p_values = {}

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label1, label2 = labels[i], labels[j]
            cluster1, cluster2 = clusters[label1], clusters[label2]

            size1, size2 = len(cluster1), len(cluster2)

            # Identify the larger and smaller clusters
            if len(cluster1) > len(cluster2):
                larger_cluster, smaller_cluster = cluster1, cluster2
            else:
                larger_cluster, smaller_cluster = cluster2, cluster1

            # Calculate mean and covariance of the larger cluster
            mean_i = np.mean(larger_cluster[:, :3], axis=0)  # Use x, y, z columns
            cov_i = np.cov(larger_cluster[:, :3], rowvar=False)
            cov_i_inv = np.linalg.inv(cov_i + np.eye(3) * 1e-5)  # Regularize covariance matrix

            # Compute Mahalanobis distances and calculate p-value for smaller cluster points
            distances = []
            for point in smaller_cluster[:, :3]:  # Use x, y, z columns
                diff = point - mean_i
                mahalanobis_square = diff @ cov_i_inv @ diff.T

                mahalanobis_distance = 0 if mahalanobis_square < 0 else np.sqrt(mahalanobis_square)
                distances.append(mahalanobis_distance)

            # Calculate the average distance and the p-value
            avg_distance = np.mean(distances)
            p_value = 0 if avg_distance <= 0 else 1 - chi2.cdf(avg_distance, df=2)

            # Store the p-value as the metric
            p_values[(label1, label2)] = (p_value, size1, size2, unique_gmm_labels)

    return p_values

# Helper function to calculate p-values for each pair of beam and track clusters
def calculate_cross_p_values(beam_clusters, track_clusters, unique_gmm_labels):
    p_values = {}

    for beam_label, beam_cluster in beam_clusters.items():
        for track_label, track_cluster in track_clusters.items():
            # Identify the larger and smaller clusters
            if len(beam_cluster) > len(track_cluster):
                larger_cluster, smaller_cluster = beam_cluster, track_cluster
            else:
                larger_cluster, smaller_cluster = track_cluster, beam_cluster

            size1, size2 = len(beam_cluster), len(track_cluster)

            # Calculate mean and covariance of the larger cluster
            mean_i = np.mean(larger_cluster[:, :3], axis=0)  # Use x, y, z columns
            cov_i = np.cov(larger_cluster[:, :3], rowvar=False)
            cov_i_inv = np.linalg.inv(cov_i + np.eye(3) * 1e-5)  # Regularize covariance matrix

            # Compute Mahalanobis distances and calculate p-value for smaller cluster points
            distances = []
            for point in smaller_cluster[:, :3]:  # Use x, y, z columns
                diff = point - mean_i
                mahalanobis_square = diff @ cov_i_inv @ diff.T

                mahalanobis_distance = 0 if mahalanobis_square < 0 else np.sqrt(mahalanobis_square)
                distances.append(mahalanobis_distance)

            # Calculate the average distance and the p-value
            avg_distance = np.mean(distances)
            p_value = 0 if avg_distance <= 0 else 1 - chi2.cdf(avg_distance, df=2)

            # Store the p-value as the metric
            p_values[(beam_label, track_label)] = (p_value, size1, size2, unique_gmm_labels)

    return p_values
