import ROOT
import glob

# Define parameters
exc_energy = [0, 5, 10, 15, 20, 25, 30]
cm_angle = [1, 2, 3, 4, 5]
files = []

# Find files matching the pattern
for ex in exc_energy:
    for cm in cm_angle:
        file_pattern = f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/reg/metrics_sim_5000_{ex}mev_{cm}cm_*.root"
        matched_files = glob.glob(file_pattern)
        files.extend(matched_files)

# Create a TChain for the events tree
chain = ROOT.TChain("events")
for file in files:
    print(f"Adding file to TChain: {file}")
    chain.Add(file)

# Create histograms for gmm_bt_metric and gmm_td_metric with updated ranges and bins
hist_bt_metric = ROOT.TH1F("hist_bt_metric", "", 100, 0, 0.7)  # No title
hist_td_metric = ROOT.TH1F("hist_td_metric", "", 100, 0, 100)  # No title

# Loop over the entries in the chain and fill the histograms
for entry in range(chain.GetEntries()):
    chain.GetEntry(entry)

    # Fill histograms by iterating over elements of the std::vector
    if chain.gmm_bt_metric:
        for value in chain.gmm_bt_metric:
            hist_bt_metric.Fill(value)

    if chain.gmm_td_metric:
        for value in chain.gmm_td_metric:
            hist_td_metric.Fill(value)

# Set the axis labels with LaTeX-style formatting
hist_bt_metric.GetXaxis().SetTitle("p_{ij}")
hist_bt_metric.GetYaxis().SetTitle("Counts")

hist_td_metric.GetXaxis().SetTitle("c_{ij}")
hist_td_metric.GetYaxis().SetTitle("Counts")

# # Set stats box to show only the mean for both histograms
# hist_bt_metric.SetStats(1)
# hist_td_metric.SetStats(1)

# # Set the stats box to display only the mean
# hist_bt_metric.GetListOfFunctions().FindObject("stats").SetOptStat("m")  # "m" for mean
# hist_td_metric.GetListOfFunctions().FindObject("stats").SetOptStat("m")  # "m" for mean

# Create a canvas with two pads
canvas = ROOT.TCanvas("canvas", "Canvas", 800, 600)

# Divide the canvas into two pads
canvas.Divide(2, 1)  # One column, two rows

# Plot the first histogram on the first pad
canvas.cd(1)  # Select pad 1
hist_bt_metric.Draw()

# Add text "(a)" in the top-left corner of the first pad
label_a = ROOT.TLatex(0.1, 0.95, "(a)")
label_a.SetNDC(True)  # Use normalized coordinates
label_a.Draw()

# Plot the second histogram on the second pad
canvas.cd(2)  # Select pad 2
hist_td_metric.Draw()

# Add text "(b)" in the top-left corner of the second pad
label_b = ROOT.TLatex(0.1, 0.95, "(b)")
label_b.SetNDC(True)  # Use normalized coordinates
label_b.Draw()

# Update the canvas to display the histograms
canvas.Update()
canvas.WaitPrimitive()  # Wait for user interaction
