import ROOT
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Define excitation energy and CM values
excitation_energies = [0, 5, 10, 15, 20, 25, 30]
cm_values = [1, 2, 3, 4, 5]

# Initialize lists to store data
distances = []
counts = {}
counter = 0

# Loop through excitation energy and CM combinations
for ex in excitation_energies:
    for cm in cm_values:
        file_path = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/multiplicity/"
        file_pattern = f"mul_sim_5000_{ex}mev_{cm}cm_*.root"
        file_list = glob.glob(file_path + file_pattern)

        print('file_pattern:', file_pattern)

        # Check if files exist before processing
        if not file_list:
            print(f"No files found for pattern: {file_pattern}")
            continue  # Skip this iteration

        chain = ROOT.TChain("events")

        # Add all files to the TChain
        for filename in file_list:
            chain.Add(filename)

        print(f"Processing {chain.GetEntries()} entries...")

        # Loop through entries
        for entry in range(chain.GetEntries()):
            chain.GetEntry(entry)

            # Get branches
            distance_vector = chain.cdist_threshold_value  # std::vector<double>
            scattered_vector = chain.cdist_threshold_scattered  # std::vector<int>

            if sum(chain.cdist_threshold_scattered) > 0:
                counter += 1

            # Loop through elements in the vectors
            for i in range(distance_vector.size()):
                distance = distance_vector[i]
                k = scattered_vector[i]  # Corresponding scattered track count

                if k == 1:
                    distances.append(distance)
                    counts[distance] = counts.get(distance, 0) + 1  # Count occurrences

# Convert to NumPy array for plotting
unique_distances = np.array(sorted(counts.keys()))
entries = np.array([counts[d] for d in unique_distances])
normalized_entries = entries / counter  # Normalize the counts

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

# First subplot: Raw counts of k=1
axes[0].bar(unique_distances, entries, width=0.5, alpha=0.7, color="b", label="Entries with k=1")
axes[0].set_xlabel("Distance")
axes[0].set_ylabel("Number of Entries")
axes[0].set_title("Entries with k=1 as a function of Distance")
axes[0].legend()
axes[0].grid()

# Second subplot: Normalized counts
axes[1].bar(unique_distances, normalized_entries, width=0.5, alpha=0.7, color="r", label="Normalized")
axes[1].set_xlabel("Distance")
axes[1].set_ylabel("Fraction of Entries")
axes[1].set_title("Normalized Entries with k=1 as a function of Distance")
axes[1].legend()
axes[1].grid()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()