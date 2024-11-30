import ROOT
from glob import glob
import numpy as np

# Define parameters
ex = 10
cm = 3
file_pattern = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/minimize/recon_sim_5000_{ex}mev_{cm}cm_*.root"
files = glob(file_pattern)

# Create a TChain for the events tree
chain = ROOT.TChain("events")
for file in files:
    print(file)
    chain.Add(file)


# Define histograms with the specified ranges and bin sizes
hist_ransac_angles = ROOT.TH1F("hist_ransac_angles", "Angles - RANSAC", 160, -20, 20)
hist_gmm_angles = ROOT.TH1F("hist_gmm_angles", "Angles - GMM", 160, -20, 20)

hist_ransac_start_x = ROOT.TH1F("hist_ransac_start_x", "Start X - RANSAC", 256, 0, 256)
hist_gmm_start_x = ROOT.TH1F("hist_gmm_start_x", "Start X - GMM", 256, 0, 256)

hist_ransac_start_y = ROOT.TH1F("hist_ransac_start_y", "Start Y - RANSAC", 256, 0, 256)
hist_gmm_start_y = ROOT.TH1F("hist_gmm_start_y", "Start Y - GMM", 256, 0, 256)

hist_ransac_start_z = ROOT.TH1F("hist_ransac_start_z", "Start Z - RANSAC", 256, 0, 256)
hist_gmm_start_z = ROOT.TH1F("hist_gmm_start_z", "Start Z - GMM", 256, 0, 256)

hist_ransac_inter_x = ROOT.TH1F("hist_ransac_inter_x", "Intersection X - RANSAC", 256, 0, 256)
hist_gmm_inter_x = ROOT.TH1F("hist_gmm_inter_x", "Intersection X - GMM", 256, 0, 256)

hist_ransac_inter_y = ROOT.TH1F("hist_ransac_inter_y", "Intersection Y - RANSAC", 256, 0, 256)
hist_gmm_inter_y = ROOT.TH1F("hist_gmm_inter_y", "Intersection Y - GMM", 256, 0, 256)

hist_ransac_inter_z = ROOT.TH1F("hist_ransac_inter_z", "Intersection Z - RANSAC", 256, 0, 256)
hist_gmm_inter_z = ROOT.TH1F("hist_gmm_inter_z", "Intersection Z - GMM", 256, 0, 256)

hist_gmm_resp = ROOT.TH1F("hist_gmm_resp", "Resp - GMM", 160, -20, 20)

# New histogram for angles corresponding to the best threshold
hist_best_angles = ROOT.TH1F("hist_best_angles", "Angles for Best Threshold", 160, -20, 20)

# Define the target angle mean (31.81) to compare against
target_angle = 60.7

# Function to calculate the mean and standard deviation of a list of values
def calculate_mean_and_sd(values):
    mean = np.mean(values)
    stddev = np.std(values)
    return mean, stddev

# Dictionary to store all angles for each threshold value
threshold_angles = {}

# Loop through the entries to fill histograms and store angles for each threshold
# Fill histograms from TChain with checks for empty arrays
for entry in chain:
    # Angles
    for angle in entry.ransac_angles:
        hist_ransac_angles.Fill(angle-target_angle)
    for angle in entry.gmm_angles:
        hist_gmm_angles.Fill(angle-target_angle)

    if len(entry.gmm_resp_angle) > 1:
        hist_best_angles.Fill(entry.gmm_resp_angle[130]-target_angle)

    # Start X, Y, Z (only fill if the array is not empty)
    if len(entry.ransac_start) >= 3:
        hist_ransac_start_x.Fill(entry.ransac_start[0])
        hist_ransac_start_y.Fill(entry.ransac_start[1])
        hist_ransac_start_z.Fill(entry.ransac_start[2])

    if len(entry.gmm_start) >= 3:
        hist_gmm_start_x.Fill(entry.gmm_start[0])
        hist_gmm_start_y.Fill(entry.gmm_start[1])
        hist_gmm_start_z.Fill(entry.gmm_start[2])

    # Intersection X, Y, Z (only fill if the array is not empty)
    if len(entry.ransac_inter) >= 3:
        hist_ransac_inter_x.Fill(entry.ransac_inter[0])
        hist_ransac_inter_y.Fill(entry.ransac_inter[1])
        hist_ransac_inter_z.Fill(entry.ransac_inter[2])

    if len(entry.gmm_inter) >= 3:
        hist_gmm_inter_x.Fill(entry.gmm_inter[0])
        hist_gmm_inter_y.Fill(entry.gmm_inter[1])
        hist_gmm_inter_z.Fill(entry.gmm_inter[2])

    for resp in entry.gmm_min_angle:
        hist_gmm_resp.Fill(resp-target_angle)

#     # For each entry, store the angle corresponding to each threshold
#     for i in range(len(entry.gmm_resp)):  # For each entry
#         threshold = entry.gmm_resp[i]       # Get threshold
#         angle = entry.gmm_resp_angle[i]     # Get corresponding angle

#         # If the threshold has not been encountered before, create a list for its angles
#         if threshold not in threshold_angles:
#             threshold_angles[threshold] = []

#         # Append the angle to the corresponding threshold's list
#         threshold_angles[threshold].append(angle)

# # Now, for each threshold, calculate the mean and standard deviation of the angles
# best_threshold = None
# best_mean_diff = float('inf')
# best_stddev = float('inf')

# # Find the threshold with the mean closest to 31.81 and the lowest standard deviation
# for threshold, angles in threshold_angles.items():
#     mean, stddev = calculate_mean_and_sd(angles)
#     # print(threshold, angles)
#     # Select the threshold with the best (closest mean) and lowest standard deviation
#     if abs(mean - target_angle) < best_mean_diff or (abs(mean - target_angle) == best_mean_diff and stddev < best_stddev):
#         best_threshold = threshold
#         best_mean_diff = abs(mean - target_angle)
#         best_stddev = stddev
#         best_angles = angles
#     print('Best Threshold', best_threshold, best_mean_diff, best_stddev)

# print(threshold_angles.keys())
# # Fill the histogram for the angles corresponding to the best threshold
# for angle in best_angles:
#     hist_best_angles.Fill(angle)


# # Plot the histograms
# def plot_histograms(pad_num, hist1, hist2, title):
#     canvas.cd(pad_num)
#     hist1.SetLineColor(ROOT.kRed)
#     hist2.SetLineColor(ROOT.kBlue)
#     hist1.SetTitle(title)
#     hist1.Draw("HIST")
#     hist2.Draw("HIST SAME")
#     hist1.SetStats(True)
#     hist2.SetStats(True)
#     legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
#     legend.AddEntry(hist1, "RANSAC", "l")
#     legend.AddEntry(hist2, "GMM", "l")
#     legend.Draw()

# # Plot the histograms in each subplot
# plot_histograms(1, hist_best_angles, hist_gmm_angles, "Angles for Best Threshold vs RANSAC")
# plot_histograms(2, hist_gmm_angles, hist_best_angles, "Angles for Best Threshold vs RANSAC")
# plot_histograms(3, hist_ransac_start_y, hist_gmm_start_y, "Start Y")
# plot_histograms(4, hist_ransac_start_z, hist_gmm_start_z, "Start Z")
# plot_histograms(5, hist_ransac_inter_x, hist_gmm_inter_x, "Intersection X")
# plot_histograms(6, hist_ransac_inter_y, hist_gmm_inter_y, "Intersection Y")
# plot_histograms(7, hist_ransac_inter_z, hist_gmm_inter_z, "Intersection Z")
# plot_histograms(8, hist_gmm_resp, hist_gmm_resp, "Resp")

# Create a ROOT canvas
canvas = ROOT.TCanvas("canvas", "7 Subplots", 1200, 900)
canvas.Divide(2, 2)
canvas.cd(1)
hist_gmm_angles.Draw()
canvas.cd(2)
hist_best_angles.Draw()
canvas.cd(3)
hist_ransac_angles.Draw()
canvas.cd(4)
hist_gmm_resp.Draw()

canvas.Update()
canvas.Draw()
canvas.WaitPrimitive()
