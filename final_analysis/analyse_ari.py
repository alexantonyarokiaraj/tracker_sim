import ROOT
import glob

# Define parameters
exc_energy = [0, 5, 10, 15, 20, 25, 30]
cm_angle = [1, 2, 3, 4, 5]
files = []

# Find files matching the pattern
for ex in exc_energy:
    for cm in cm_angle:
        file_pattern = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/multiplicity/full/mul_sim_5000_{ex}mev_{cm}cm_*.root"
        matched_files = glob.glob(file_pattern)
        files.extend(matched_files)

# Create a TChain for the events tree
chain = ROOT.TChain("events")
for file in files:
    print(f"Adding file to TChain: {file}")
    chain.Add(file)

# Check the number of entries in the chain
total_entries = chain.GetEntries()
if total_entries == 0:
    print("No entries found in the TChain. Exiting.")
    exit(1)
print(f"Total entries in the chain: {total_entries}")

# Check if all required branches exist
required_branches = ["ransac_ari", "gmm_ari_pval", "gmm_ari_cdist"]
for branch in required_branches:
    if not chain.GetListOfBranches().FindObject(branch):
        print(f"Branch '{branch}' not found in the tree. Exiting.")
        exit(1)

# Create histograms
hist_ransac = ROOT.TH1F("ransac_ari", "", 60, 0.5, 1.2)
hist_gmm_pval = ROOT.TH1F("gmm_ari_pval", "", 60, 0.5, 1.2)
hist_gmm_cdist = ROOT.TH1F("gmm_ari_cdist", "", 60, 0.5, 1.2)

# Loop through the entries in the tree
for i in range(total_entries):
    chain.GetEntry(i)  # Load the i-th entry into memory

    # Fill histograms
    for branch, hist in zip(required_branches, [hist_ransac, hist_gmm_pval, hist_gmm_cdist]):
        metric_vector = getattr(chain, branch, None)
        if metric_vector:  # Check if the branch contains valid data
            for value in metric_vector:
                hist.Fill(value)

# Calculate means
mean_ransac = hist_ransac.GetMean()
mean_gmm_pval = hist_gmm_pval.GetMean()
mean_gmm_cdist = hist_gmm_cdist.GetMean()

# Determine the maximum bin content for proper scaling
max_y = max(hist_ransac.GetMaximum(), hist_gmm_pval.GetMaximum(), hist_gmm_cdist.GetMaximum())

# Create a Canvas
canvas = ROOT.TCanvas("canvas", "Comparison of Metrics", 800, 600)

# Set histogram colors
hist_ransac.SetLineColor(ROOT.kBlue)
hist_gmm_pval.SetLineColor(ROOT.kRed)
hist_gmm_cdist.SetLineColor(ROOT.kGreen)

# Set the maximum Y-axis limit
hist_ransac.SetMaximum(max_y * 1.1)  # Add 10% margin to the highest value

# Draw histograms
hist_ransac.Draw("HIST")
hist_gmm_pval.Draw("HIST SAME")
hist_gmm_cdist.Draw("HIST SAME")

# Set the X and Y axis labels
# Set the X and Y axis labels with proper spacing
hist_ransac.GetXaxis().SetTitle(r"Adjusted\ Rand\ Index\ (\text{ARI})")  # Added explicit spaces using "\ "
hist_ransac.GetYaxis().SetTitle(r"Counts")  # Y-axis label

# Disable statistics box
hist_ransac.SetStats(0)

# Add a legend with LaTeX formatting
legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
legend.SetTextFont(42)  # Use the default font for LaTeX
legend.SetTextSize(0.03)  # Adjust the text size if needed

# Set the legend entries with LaTeX (no `str.format()` here)
legend.AddEntry(hist_ransac, r"\text{RANSAC}" + ", mean = {:.2f}".format(mean_ransac), "l")
legend.AddEntry(hist_gmm_pval, r"\mathcal{R}_{pij}" + ", mean = {:.2f}".format(mean_gmm_pval), "l")
legend.AddEntry(hist_gmm_cdist, r"\mathcal{R}_{pij, cij}" + ", mean = {:.2f}".format(mean_gmm_cdist), "l")


# Draw the legend
legend.Draw()

# Update and display the canvas
canvas.Update()
canvas.WaitPrimitive()
