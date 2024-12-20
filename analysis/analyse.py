import ROOT
import os
import numpy as np

# Directory containing the ROOT files
file_dir = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/responsibilities/"

# Define excitation energies and cm angles
excitation_energies = [0, 5, 10, 15, 20, 25, 30]
cm_angles = [1, 2, 3, 4, 5]

# Initialize histograms
h_bb_metric = ROOT.TH1F("h_bb_metric", "Histogram of GMM BB Metric;Metric Value;Counts", 10000, 0, 1)
h_bt_metric = ROOT.TH1F("h_bt_metric", "Histogram of GMM BT Metric;Metric Value;Counts", 1000, 0, 1)
h_tt_metric = ROOT.TH1F("h_tt_metric", "Histogram of GMM TT Metric;Metric Value;Counts", 1000, 0, 1)

# Loop over all combinations of excitation energies and cm angles
for ex in excitation_energies:
    for cm in cm_angles:
        # Construct the file name
        file_name = f"recon_sim_5000_{ex}mev_{cm}cm_1_5000.root"
        file_path = os.path.join(file_dir, file_name)

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Open the ROOT file and get the tree
        root_file = ROOT.TFile.Open(file_path)
        tree = root_file.Get("events")
        if not tree:
            print(f"Tree 'events' not found in file: {file_path}")
            root_file.Close()
            continue

        # Loop over entries in the tree
        for entry in tree:
            # Add metric values to histograms
            for value in entry.gmm_bb_metric:
                h_bb_metric.Fill(value)
            for value in entry.gmm_bt_metric:
                h_bt_metric.Fill(value)
            for value in entry.gmm_tt_metric:
                h_tt_metric.Fill(value)

        root_file.Close()

# Save histograms to an output ROOT file
output_file = ROOT.TFile("/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/histograms/metric_histograms.root", "RECREATE")
h_bb_metric.Write()
h_bt_metric.Write()
h_tt_metric.Write()
output_file.Close()

print("Histograms created and saved to 'metric_histograms.root'")
