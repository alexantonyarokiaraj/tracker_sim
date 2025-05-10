import ROOT
import glob
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# Settings
excitation_energies = [10]
cm_angles = [1, 2, 3, 4, 5]
base_path = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/final/gamma_2_run/"
volume_min, volume_max = 10, 246
beam_zone_min, beam_zone_max = 120, 134
hist_range = (-20, 20)
n_bins = 80

# Responsibility binning
res = np.linspace(0, 1.0, 5000)  # 5000 uniform-width bins

# Helper functions
def is_inside_volume(point):
    return all(volume_min <= coord <= volume_max for coord in point)

def is_outside_beam_zone(y):
    return not (beam_zone_min <= y <= beam_zone_max)

def extract_1track_vector(vec):
    return [vec[0], vec[1], vec[2]]

def process_file_gmm_resp(filepath, res, input_angle, branch_resp, branch_resp_angle,
                          branch_start, branch_end, branch_inter, branch_phi):
    resp_bins = defaultdict(list)
    seen_event_bin_pairs = set()

    file = ROOT.TFile.Open(filepath)
    if not file or file.IsZombie():
        print(f"Could not open {filepath}")
        return resp_bins

    tree = file.Get("events")
    if not tree:
        print(f"No 'events' tree in {filepath}")
        file.Close()
        return resp_bins

    for i_event, event in enumerate(tree):
        elab = getattr(event, input_angle)
        sim_angle = elab[0]

        phi_angles = getattr(event, branch_phi)
        if len(phi_angles) != 1:
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

        responsibilities = getattr(event, branch_resp)
        angles = getattr(event, branch_resp_angle)

        # print(responsibilities)

        # if len(responsibilities) != len(angles):
        #     continue

        for resp_val, reco_angle in zip(responsibilities, angles):
            diff = sim_angle - reco_angle
            if hist_range[0] < diff < hist_range[1]:
                idx = np.searchsorted(res, resp_val, side="right") - 1
                if 0 <= idx < len(res) - 1:
                    event_bin_key = (i_event, idx)
                    if event_bin_key in seen_event_bin_pairs:
                        # print(f"Deduplicating event {i_event}, bin index {idx} (resp = {resp_val:.6f})")
                        continue
                    seen_event_bin_pairs.add(event_bin_key)
                    resp_bins[idx].append(diff)

    file.Close()
    return resp_bins


# Assuming res, n_bins, hist_range are already defined elsewhere in your script

array_save_list = []

# Loop through energies and CM angles to process data
for energy in excitation_energies:
    for cm in cm_angles:
        resp_diff_map = defaultdict(list)
        # Pattern to match files
        pattern = os.path.join(base_path, f"gamma_sim_5000_{energy}mev_{cm}cm_*_*.root")
        file_list = glob.glob(pattern)

        for filepath in file_list:
            bins = process_file_gmm_resp(filepath, res, "Elab",
                                         "gmm_resp", "gmm_resp_angle",
                                         "gmm_start", "gmm_end", "gmm_inter",
                                         "gmm_phi_angles")
            for idx, diffs in bins.items():
                resp_diff_map[idx].extend(diffs)

        print(f"\nResults for {energy} MeV, {cm}°:")
        for idx in sorted(resp_diff_map.keys()):
            diffs = resp_diff_map[idx]
            if len(diffs) < 5:
                continue  # Skip if there are too few data points

            bin_low = res[idx]
            bin_high = res[idx + 1] if idx + 1 < len(res) else res[idx] + 0.01
            hist = ROOT.TH1F(f"hist_resp_{energy}_{cm}_{idx}", "", n_bins, hist_range[0], hist_range[1])
            for d in diffs:
                hist.Fill(d)

            # Compute mean and standard deviation for this bin
            mean = hist.GetMean()
            sigma = hist.GetStdDev()
            print(f"Resp [{bin_low:.4f}, {bin_high:.4f}): N={len(diffs)}, Mean={mean:.4f}, Sigma={sigma:.4f}")
            array_save_list.append([energy, cm, bin_low, bin_high, len(diffs), mean, sigma])

    array_save = np.array(array_save_list)
    np.save("angle_diff_resp.npy", array_save)
