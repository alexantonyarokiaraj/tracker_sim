import ROOT
import glob
import numpy as np
import os
from collections import defaultdict

# Settings
excitation_energies = [10]
cm_angles = [1, 2, 3, 4, 5]
base_path = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/final/beta/"
volume_min, volume_max = 10, 246
beam_zone_min, beam_zone_max = 120, 134
hist_range = (-20, 20)
n_bins = 80
beta_bin_width = 0.01

def is_inside_volume(point):
    return all(volume_min <= coord <= volume_max for coord in point)

def is_outside_beam_zone(y):
    return not (beam_zone_min <= y <= beam_zone_max)

def extract_1track_vector(vec):
    return [vec[0], vec[1], vec[2]]

def process_file(filepath, branch_angle, branch_phi, branch_start, branch_end, branch_inter,
                 input_angle, branch_beta, branch_beta_angle):

    beta_groups = defaultdict(list)

    file = ROOT.TFile.Open(filepath)
    if not file or file.IsZombie():
        print(f"Could not open {filepath}")
        return beta_groups

    tree = file.Get("events")
    if not tree:
        print(f"No 'events' tree in {filepath}")
        return beta_groups

    for event in tree:
        elab = getattr(event, input_angle)
        sim_angle = elab[0]

        reco_angles = getattr(event, branch_angle)
        phi_angles = getattr(event, branch_phi)

        if len(reco_angles) != 1 or len(phi_angles) != 1:
            continue

        phi = phi_angles[0]
        if 70 <= abs(phi) <= 110:
            continue

        start = extract_1track_vector(getattr(event, branch_start))
        end = extract_1track_vector(getattr(event, branch_end))
        inter = extract_1track_vector(getattr(event, branch_inter))

        if not (is_inside_volume(start) and is_inside_volume(end) and is_inside_volume(inter)):
            continue
        if not is_outside_beam_zone(end[1]):
            continue

        betas = getattr(event, branch_beta)
        beta_angles = getattr(event, branch_beta_angle)

        for beta, beta_angle in zip(betas, beta_angles):
            diff = beta_angle - sim_angle
            if hist_range[0] < diff < hist_range[1]:
                beta_bin = round(beta / beta_bin_width) * beta_bin_width
                beta_groups[beta_bin].append(diff)

    file.Close()
    return beta_groups

# Helper to generate sigma vs beta graph
def beta_sigma_graph(beta_diff_map, label, color, energy, cm):
    graph = ROOT.TGraph()
    i = 0
    for beta in sorted(beta_diff_map.keys()):
        diffs = beta_diff_map[beta]
        if len(diffs) < 5:
            continue
        hist = ROOT.TH1F(f"temp_{label}_{i}", "", n_bins, hist_range[0], hist_range[1])
        for d in diffs:
            hist.Fill(d)
        sigma = hist.GetStdDev()
        graph.SetPoint(i, beta, sigma)
        i += 1
    graph.SetTitle(f"σ(Δθ) vs β - {label} | {energy}MeV {cm}°;β;σ(Δθ)")
    graph.SetMarkerColor(color)
    graph.SetLineColor(color)
    graph.SetMarkerStyle(20)
    return graph

# Main loop
for energy in excitation_energies:
    all_gmm_data = []
    for cm in cm_angles:
        ransac_beta_diff_map = defaultdict(list)
        gmm_beta_diff_map = defaultdict(list)

        pattern = os.path.join(base_path, f"beta_sim_5000_{energy}mev_{cm}cm_*_*.root")
        file_list = glob.glob(pattern)

        for filepath in file_list:
            ransac_groups = process_file(filepath,
                "ransac_angles", "ransac_phi_angles",
                "ransac_start", "ransac_end", "ransac_inter",
                "Elab", "beta_ransac", "beta_ransac_angle")

            gmm_groups = process_file(filepath,
                "gmm_angles", "gmm_phi_angles",
                "gmm_start", "gmm_end", "gmm_inter",
                "Elab", "beta_gmm", "beta_gmm_angle")

            for beta, diffs in ransac_groups.items():
                ransac_beta_diff_map[beta].extend(diffs)
            for beta, diffs in gmm_groups.items():
                gmm_beta_diff_map[beta].extend(diffs)

        # Plotting
        gr_ransac = beta_sigma_graph(ransac_beta_diff_map, "RANSAC", ROOT.kBlue, energy, cm)
        gr_gmm = beta_sigma_graph(gmm_beta_diff_map, "GMM", ROOT.kRed, energy, cm)

        x = ROOT.Double()
        y = ROOT.Double()
        for i in range(gr_gmm.GetN()):
            gr_gmm.GetPoint(i, x, y)
            all_gmm_data.append([cm, float(x), float(y)])

        canvas = ROOT.TCanvas(f"c_sigma_{energy}_{cm}", "", 800, 600)
        mg = ROOT.TMultiGraph()
        mg.Add(gr_ransac, "LP")
        mg.Add(gr_gmm, "LP")
        mg.SetTitle(f"σ(Δθ) vs β — {energy} MeV, CM {cm}°;β;σ(Δθ)")
        mg.Draw("A")

        legend = ROOT.TLegend(0.7, 0.75, 0.9, 0.9)
        legend.AddEntry(gr_ransac, "RANSAC", "lp")
        legend.AddEntry(gr_gmm, "GMM", "lp")
        legend.Draw()

        canvas.Update()
        # canvas.SaveAs(f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta/histograms/sigma_vs_beta_{energy}MeV_{cm}cm.png")
        # canvas.WaitPrimitive()

    # Convert and save the combined data
    gmm_array = np.array(all_gmm_data)

    output_array_dir = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta/histograms/"
    os.makedirs(output_array_dir, exist_ok=True)

    np.save(os.path.join(output_array_dir, f"gmm_sigma_vs_beta_all_angles_{energy}MeV.npy"), gmm_array)
