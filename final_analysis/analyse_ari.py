import ROOT
import glob

# ----------------------------
# Define parameters
# ----------------------------
exc_energy = [10]
cm_angle = [1, 2, 3, 4, 5]
files = []

# Find files matching the pattern
for ex in exc_energy:
    for cm in cm_angle:
        file_pattern = f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/root_files_2/final_sim_5000_{ex}mev_{cm}cm_*.root"
        matched_files = glob.glob(file_pattern)
        files.extend(matched_files)

if not files:
    print("⚠️ No ROOT files matched the pattern. Exiting.")
    exit(1)

# ----------------------------
# Create a TChain for the events tree
# ----------------------------
chain = ROOT.TChain("events")
for file in files:
    print(f"Adding file to TChain: {file}")
    chain.Add(file)

# Check the number of entries
total_entries = chain.GetEntries()
if total_entries == 0:
    print("No entries found in the TChain. Exiting.")
    exit(1)
print(f"Total entries in the chain: {total_entries}")

# ----------------------------
# Required branches
# ----------------------------
required_branches = [
    "ransac_ari",
    "gmm_ari_pval",
    "gmm_ari_cdist",
    "ransac_filtered_ari",
    "gmm_ari"
]

for branch in required_branches:
    if not chain.GetListOfBranches().FindObject(branch):
        print(f"Branch '{branch}' not found in the tree. Exiting.")
        exit(1)

# ----------------------------
# Create histograms
# ----------------------------

bin_width = ((1.2-0.5)/50)
x_min, x_max = 0.0, 1.2
n_bins = int((x_max - x_min) / bin_width)

hist_ransac = ROOT.TH1F("ransac_ari", "", n_bins, x_min, x_max)
hist_gmm_pval = ROOT.TH1F("gmm_ari_pval", "", n_bins, x_min, x_max)
hist_gmm_cdist = ROOT.TH1F("gmm_ari_cdist", "", n_bins, x_min, x_max)
hist_ransac_filtered = ROOT.TH1F("ransac_filtered_ari", "", n_bins, x_min, x_max)
hist_gmm = ROOT.TH1F("gmm_ari", "", n_bins, x_min, x_max)

branch_to_hist = {
    "ransac_ari": hist_ransac,
    "gmm_ari_pval": hist_gmm_pval,
    "gmm_ari_cdist": hist_gmm_cdist,
    "ransac_filtered_ari": hist_ransac_filtered,
    "gmm_ari": hist_gmm,
}

# ----------------------------
# Fill histograms
# ----------------------------
for i in range(total_entries):
    chain.GetEntry(i)
    for branch, hist in branch_to_hist.items():
        metric_vector = getattr(chain, branch, None)
        if metric_vector:
            for value in metric_vector:
                hist.Fill(value)

# ----------------------------
# Calculate means
# ----------------------------
mean_ransac = hist_ransac.GetMean()
mean_gmm_pval = hist_gmm_pval.GetMean()
mean_gmm_cdist = hist_gmm_cdist.GetMean()
mean_ransac_filtered = hist_ransac_filtered.GetMean()
mean_gmm = hist_gmm.GetMean()

# ----------------------------
# Drawing
# ----------------------------
canvas = ROOT.TCanvas("canvas", "Comparison of Metrics", 800, 600)

# Assign colors
hist_ransac.SetLineColor(ROOT.kBlue)
hist_gmm_pval.SetLineColor(ROOT.kRed)
hist_gmm_cdist.SetLineColor(ROOT.kGreen)
hist_ransac_filtered.SetLineColor(ROOT.kMagenta)
hist_gmm.SetLineColor(ROOT.kOrange)

# Different line styles (to help distinguish in B/W printing)
hist_ransac.SetLineStyle(1)          # solid
hist_gmm_pval.SetLineStyle(2)        # dashed
hist_gmm_cdist.SetLineStyle(3)       # dotted
hist_ransac_filtered.SetLineStyle(4) # dash-dot
hist_gmm.SetLineStyle(5)             # long dash

# Scale Y-axis
max_y = max(h.GetMaximum() for h in branch_to_hist.values())
hist_ransac.SetMaximum(max_y * 1.1)

# Draw histograms
hist_ransac.Draw("HIST")
hist_gmm_pval.Draw("HIST SAME")
# hist_gmm_cdist.Draw("HIST SAME")
# hist_ransac_filtered.Draw("HIST SAME")
# hist_gmm.Draw("HIST SAME")

# Axis labels
hist_ransac.GetXaxis().SetTitle("Adjusted Rand Index (ARI)")
hist_ransac.GetYaxis().SetTitle("Counts")
hist_ransac.SetStats(0)

# Entries
entries_ransac = hist_ransac.GetEntries()
entries_gmm_pval = hist_gmm_pval.GetEntries()
entries_gmm_cdist = hist_gmm_cdist.GetEntries()
entries_ransac_filtered = hist_ransac_filtered.GetEntries()
entries_gmm = hist_gmm.GetEntries()

print("\nHistogram statistics:")
print(f"  RANSAC              → mean = {mean_ransac:.4f}, entries = {entries_ransac}")
print(f"  GMM (pval)          → mean = {mean_gmm_pval:.4f}, entries = {entries_gmm_pval}")
print(f"  GMM (cdist)         → mean = {mean_gmm_cdist:.4f}, entries = {entries_gmm_cdist}")
print(f"  RANSAC (filtered)   → mean = {mean_ransac_filtered:.4f}, entries = {entries_ransac_filtered}")
print(f"  GMM                 → mean = {mean_gmm:.4f}, entries = {entries_gmm}")

# Legend
legend = ROOT.TLegend(0.55, 0.55, 0.9, 0.9)
legend.AddEntry(hist_ransac, f"RANSAC", "l")
legend.AddEntry(hist_gmm_pval, f"GMM regularized by #it{{p}}_{{kl}}", "l")
# legend.AddEntry(hist_gmm_cdist, f"#it{{R}}_{{pij,cij}}, mean = {mean_gmm_cdist:.2f}", "l")
# legend.AddEntry(hist_ransac_filtered, f"RANSAC (filtered), mean = {mean_ransac_filtered:.2f}", "l")
# legend.AddEntry(hist_gmm, f"GMM, mean = {mean_gmm:.2f}", "l")
legend.Draw()

# Save and update
canvas.Update()
# canvas.SaveAs("metric_comparison.png")
canvas.WaitPrimitive()