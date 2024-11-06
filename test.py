import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

# Load the data from the .npy file
data = filtered_data  # Replace with the actual .npy data loading if needed

# Extract clusters and coordinates
clusters = data[:, 0]  # cluster labels
xyz_data = data[:, 1:4]  # [x, y, z] coordinates

# Get unique clusters
unique_clusters = np.unique(clusters)

def calculate_g_matrix(data, clusters, threshold=0.1):
    unique_clusters = np.unique(clusters)
    K = len(unique_clusters)

    # Initialize the G matrix
    G = np.zeros((K, K))

    # Calculate means and covariances for each cluster
    means = {}
    cov_matrices = {}
    counts = {}

    for cluster in unique_clusters:
        cluster_data = xyz_data[clusters == cluster]
        means[cluster] = np.mean(cluster_data, axis=0)
        cov_matrices[cluster] = np.cov(cluster_data.T)
        counts[cluster] = cluster_data.shape[0]

    # Fill the G matrix
    for i in range(K):
        for j in range(i + 1, K):  # Only need to calculate upper triangle
            cluster_i = unique_clusters[i]
            cluster_j = unique_clusters[j]

            # Decide which cluster to use as reference (the one with more points)
            if counts[cluster_i] > counts[cluster_j]:
                ci, cj = cluster_i, cluster_j
            elif counts[cluster_i] < counts[cluster_j]:
                ci, cj = cluster_j, cluster_i
            else:
                ci, cj = min(cluster_i, cluster_j), max(cluster_i, cluster_j)

            # Calculate the Mahalanobis distances
            data_j = xyz_data[clusters == cj]  # Points from the smaller cluster (cj)
            mean_i = means[ci]  # Mean of the larger cluster (ci)

            try:
                cov_i_inv = np.linalg.inv(cov_matrices[ci])  # Covariance matrix of the larger cluster (ci)

                # Compute the distance for each point in cluster j relative to cluster i
                distances = []
                for point in data_j:
                    diff = point - mean_i
                    mahalanobis_square = diff @ cov_i_inv @ diff.T

                    if mahalanobis_square < 0:
                        mahalanobis_distance = 0
                    else:
                        mahalanobis_distance = np.sqrt(mahalanobis_square)

                    distances.append(mahalanobis_distance)

                avg_distance = np.mean(distances)

                if avg_distance <= 0:
                    p_value = 0
                else:
                    p_value = 1 - chi2.cdf(avg_distance, df=2)

                # Apply threshold
                if p_value < threshold:
                    p_value = 0

                G[i, j] = round(p_value, 4)
                G[j, i] = G[i, j]  # Ensuring symmetry

            except np.linalg.LinAlgError:
                G[i, j] = 0
                G[j, i] = G[i, j]  # Ensure symmetry here as well
                print(f'Singular covariance matrix encountered for clusters ({ci}, {cj}). Setting G[{i},{j}] and G[{j},{i}] to 0.')

    # Ensure only the maximum value in each column remains non-zero
    for j in range(K):
        non_zero_values = G[:, j][G[:, j] != 0]
        if non_zero_values.size > 0:
            max_value = np.max(non_zero_values)
            for i in range(K):
                if G[i, j] != max_value:
                    G[i, j] = 0
                    G[j, i] = 0

    np.fill_diagonal(G, 0)

    return G

# Main merging loop
iteration = 0
while True:
    iteration += 1
    # Calculate the G matrix
    G = calculate_g_matrix(xyz_data, clusters)

    # Visualize the G matrix during each iteration
    plt.figure(figsize=(16, 14))  # Increase the figure size
    sns.heatmap(G, annot=True, fmt='.4f', cmap='viridis', cbar=True,
                annot_kws={"size": 10}, linewidths=.5, linecolor='black')

    plt.xticks(ticks=np.arange(len(unique_clusters)) + 0.5, labels=unique_clusters, fontsize=12)
    plt.yticks(ticks=np.arange(len(unique_clusters)) + 0.5, labels=unique_clusters, fontsize=12)

    plt.title(f'G Matrix Visualization - Iteration {iteration}', fontsize=16)
    plt.xlabel('Cluster Index (j)', fontsize=14)
    plt.ylabel('Cluster Index (i)', fontsize=14)
    plt.show()

    # Check for non-zero elements in G to decide whether to merge
    non_zero_indices = np.argwhere(G != 0)

    if non_zero_indices.size == 0:  # Exit if no non-zero elements
        print("No non-zero elements found. Exiting loop.")
        break

    # Create a mapping for merged clusters
    cluster_mapping = {label: label for label in unique_clusters}

    # Merge clusters for every non-zero element p_{ij}
    for i, j in non_zero_indices:
        if i < j:  # Ensure each pair is only processed once
            print(f"Merging clusters {unique_clusters[i]} and {unique_clusters[j]}.")

            # Merge clusters
            merged_cluster_label = max(unique_clusters[i], unique_clusters[j])  # Keep the higher label for consistency
            combined_data = np.concatenate((xyz_data[clusters == unique_clusters[i]],
                                             xyz_data[clusters == unique_clusters[j]]))

            # Update cluster labels using mapping
            clusters[clusters == unique_clusters[j]] = merged_cluster_label  # Merge cluster j into i
            cluster_mapping[unique_clusters[i]] = merged_cluster_label
            cluster_mapping[unique_clusters[j]] = merged_cluster_label

            # Update clusters based on the mapping
            for old_label, new_label in cluster_mapping.items():
                clusters[clusters == old_label] = new_label

            # Update unique_clusters list
            unique_clusters = np.unique(clusters)

    # Scatter plot of the data points color-coded by their new cluster labels
    plt.figure(figsize=(10, 8))
    for label in unique_clusters:
        cluster_data = xyz_data[clusters == label]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {label}')

        # Calculate mean position for labeling
        mean_position = np.mean(cluster_data, axis=0)
        plt.text(mean_position[0], mean_position[1], str(label), fontsize=12,
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    plt.title(f'Scatter Plot of Data Points - Iteration {iteration}', fontsize=16)
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

print("Final G matrix:")
print(G)
