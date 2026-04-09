import ROOT
import glob
import numpy as np
import os

# Settings
excitation_energies = [10]
cm_angles = [1]
base_path = "/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/new/compare"
volume_min, volume_max = 10, 246
beam_zone_min, beam_zone_max = 120, 134
hist_range = (-50, 50)
n_bins = 400

def is_inside_volume(point):
    return all(volume_min <= coord <= volume_max for coord in point)

def is_outside_beam_zone(y):
    return not (beam_zone_min <= y <= beam_zone_max)

def extract_1track_vector(vec):
    return [vec[0], vec[1], vec[2]]

def process_file(filepath, branch_angle, branch_phi, branch_start, branch_end, branch_inter, angle_type):
    diff_angles = []

    file = ROOT.TFile.Open(filepath)
    if not file or file.IsZombie():
        print(f"Could not open {filepath}")
        return diff_angles

    tree = file.Get("events")
    if not tree:
        print(f"No 'events' tree in {filepath}")
        return diff_angles

    for event in tree:
        elab = getattr(event, angle_type)
        reco_angles = getattr(event, branch_angle)
        phi_angles = getattr(event, branch_phi)

        sim_angle = elab[0]

        if len(reco_angles) != 1 or len(phi_angles) != 1:
            continue

        phi = phi_angles[0]
        if (70 <= abs(phi) <= 110):
            continue  # discard based on phi angle

        start = extract_1track_vector(getattr(event, branch_start))
        end = extract_1track_vector(getattr(event, branch_end))
        inter = extract_1track_vector(getattr(event, branch_inter))

        if not (is_inside_volume(start) and is_inside_volume(end) and is_inside_volume(inter)):
            continue

        if not is_outside_beam_zone(end[1]):
            continue

        angle_diff = reco_angles[0] - sim_angle
        if hist_range[0] < angle_diff < hist_range[1]:
            diff_angles.append(angle_diff)

    file.Close()
    return diff_angles


# Main loop
for energy in excitation_energies:
    for cm in cm_angles:
        ransac_diffs = []
        gmm_diffs = []

        pattern = os.path.join(base_path, f"compare_sim_5000_{energy}mev_{cm}cm_*_*.root")
        file_list = glob.glob(pattern)

        for filepath in file_list:
            ransac_diffs += process_file(filepath, "ransac_angles", "ransac_phi_angles",
                                         "ransac_start", "ransac_end", "ransac_inter", "Elab")
            gmm_diffs += process_file(filepath, "gmm_angles", "gmm_phi_angles",
                                      "gmm_start", "gmm_end", "gmm_inter", "Elab")

        # Create canvas and histograms
        canvas = ROOT.TCanvas(f"c_{energy}_{cm}", f"Energy {energy} MeV, CM {cm}°", 1200, 600)
        canvas.Divide(2, 1)

        h_ransac = ROOT.TH1F(f"ransac_hist_{energy}_{cm}", f"RANSAC Δθ - Energy {energy} MeV, CM {cm}°;Δθ (deg);Entries", n_bins, hist_range[0], hist_range[1])
        h_gmm = ROOT.TH1F(f"gmm_hist_{energy}_{cm}", f"GMM Δθ - Energy {energy} MeV, CM {cm}°;Δθ (deg);Entries", n_bins, hist_range[0], hist_range[1])

        for diff in ransac_diffs:
            h_ransac.Fill(diff)
        for diff in gmm_diffs:
            h_gmm.Fill(diff)

        # Stats
        def annotate_hist(hist):
            entries = hist.GetEntries()
            mean = hist.GetMean()
            stddev = hist.GetStdDev()
            hist.SetStats(0)
            label = ROOT.TLatex()
            label.SetNDC()
            label.SetTextSize(0.03)
            label.DrawLatex(0.15, 0.85, f"Entries: {int(entries)}")
            label.DrawLatex(0.15, 0.80, f"Mean: {mean:.2f}")
            label.DrawLatex(0.15, 0.75, f"Std Dev: {stddev:.2f}")

        canvas.cd(1)
        h_ransac.SetLineColor(ROOT.kBlue)
        h_ransac.Draw()
        annotate_hist(h_ransac)

        canvas.cd(2)
        h_gmm.SetLineColor(ROOT.kRed)
        h_gmm.Draw()
        annotate_hist(h_gmm)

        canvas.Update()
        canvas.SaveAs(f"histogram_{energy}MeV_{cm}cm.png")
        canvas.WaitPrimitive()