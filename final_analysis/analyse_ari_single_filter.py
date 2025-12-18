import ROOT
import glob

# -----------------------------
# Settings
# -----------------------------
excitation_energies = [10]
cm_angles = [5]
suppression_factor = [1]
base_path = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/root_files_1/"
volume_min, volume_max = 10, 246
beam_zone_min, beam_zone_max = 120, 134
hist_range = (-20, 20)
n_bins_ari = 50
ari_min, ari_max = 0.0, 1.2

# -----------------------------
# Helper functions
# -----------------------------
def is_inside_volume(point):
    return all(volume_min <= coord <= volume_max for coord in point)

def is_outside_beam_zone(y):
    return not (beam_zone_min <= y <= beam_zone_max)

def extract_1track_vector(vec):
    return [vec[0], vec[1], vec[2]]

def filter_event(event, branch_angle, branch_phi, branch_start, branch_end, branch_inter, input_angle):
    reasons = []

    reco_angles = getattr(event, branch_angle)
    phi_angles = getattr(event, branch_phi)

    if len(reco_angles) == 0:
        reasons.append("no_track_found")
    elif len(reco_angles) > 1:
        reasons.append("more_than_1_track")
    else:
        phi = phi_angles[0]
        if 70 <= abs(phi) <= 110:
            reasons.append("phi_cut")

        start = extract_1track_vector(getattr(event, branch_start))
        end = extract_1track_vector(getattr(event, branch_end))
        inter = extract_1track_vector(getattr(event, branch_inter))

        if not is_inside_volume(start):
            reasons.append("start_outside")
        if not is_inside_volume(end):
            reasons.append("end_outside")
        if not is_inside_volume(inter):
            reasons.append("inter_outside")
        if not is_outside_beam_zone(end[1]):
            reasons.append("end_in_beam_zone")

        sim_angle = getattr(event, input_angle)[0]
        angle_diff = sim_angle - reco_angles[0]
        if not (hist_range[0] < angle_diff < hist_range[1]):
            reasons.append("angle_diff_cut")

    accepted = len(reasons) == 0
    return accepted, reasons

# -----------------------------
# Collect ARI values for all branches
# Only for GMM-rejected events due to "more_than_1_track"
# -----------------------------
required_branches = [
    "ransac_ari",
    "ransac_filtered_ari",
    "gmm_ari",
    "gmm_ari_pval",
    "gmm_ari_cdist"
]

ari_values = {branch: [] for branch in required_branches}

for energy in excitation_energies:
    for cm in cm_angles:
        for suppress in suppression_factor:
            pattern = f"{base_path}/final_sim_5000_{energy}mev_{cm}cm_*_*_{suppress}.root"
            file_list = glob.glob(pattern)

            for filepath in file_list:
                file = ROOT.TFile.Open(filepath)
                if not file or file.IsZombie():
                    continue
                tree = file.Get("events")
                if not tree:
                    continue

                for event in tree:
                    accepted, reasons = filter_event(event,
                                                     branch_angle="gmm_angles",
                                                     branch_phi="gmm_phi_angles",
                                                     branch_start="gmm_start",
                                                     branch_end="gmm_end",
                                                     branch_inter="gmm_inter",
                                                     input_angle="Elab")
                    if "more_than_1_track" in reasons:
                        for branch in required_branches:
                            value = getattr(event, branch)
                            if len(value) > 0:
                                ari_values[branch].append(value[0])

                file.Close()

# -----------------------------
# Create Canvas and Histograms
# -----------------------------
canvas = ROOT.TCanvas("c_gmm_rejected_all_ari", "GMM-rejected ARI (more than 1 track)", 1000, 700)

colors = [ROOT.kBlue, ROOT.kGreen+2, ROOT.kRed, ROOT.kMagenta, ROOT.kOrange]
histograms = {}

for i, branch in enumerate(required_branches):
    hist = ROOT.TH1F(f"hist_{branch}", f"{branch};ARI;Counts", n_bins_ari, ari_min, ari_max)
    for val in ari_values[branch]:
        hist.Fill(val)
    hist.SetLineColor(colors[i])
    hist.SetLineWidth(2)
    histograms[branch] = hist

# Draw all histograms on the same pad
histograms[required_branches[0]].Draw("HIST")
for branch in required_branches[1:]:
    histograms[branch].Draw("HIST SAME")

# Legend
legend = ROOT.TLegend(0.55, 0.55, 0.9, 0.9)
for branch, hist in histograms.items():
    legend.AddEntry(hist, f"{branch} (n={len(ari_values[branch])})", "l")
legend.Draw()

canvas.Update()
canvas.SaveAs("gmm_rejected_more_than_1_track_all_ARI.png")
canvas.WaitPrimitive()
