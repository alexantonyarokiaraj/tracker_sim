import ROOT
import glob
import numpy as np
import os
import pandas as pd
from collections import defaultdict
# from upsetplot import from_indicators, plot as upset_plot
import matplotlib.pyplot as plt

# Settings
excitation_energies = [10]
cm_angles = [5]
base_path = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/root_files_2/"
volume_min, volume_max = 10, 246
beam_zone_min, beam_zone_max = 120, 134
hist_range = (-20, 20)

filter_names = [
    "no_track_found",
    "more_than_1_track",
    "phi_cut",
    "start_outside",
    "end_outside",
    "inter_outside",
    "end_in_beam_zone",
    "angle_diff_cut"
]

# Helper functions
def is_inside_volume(point):
    return all(volume_min <= coord <= volume_max for coord in point)

def is_outside_beam_zone(y):
    return not (beam_zone_min <= y <= beam_zone_max)

def extract_1track_vector(vec):
    return [vec[0], vec[1], vec[2]]

# Main processing function
def process_file(filepath, branch_angle, branch_phi, branch_start, branch_end, branch_inter, input_angle, method_label, reason_tracker):
    file = ROOT.TFile.Open(filepath)
    if not file or file.IsZombie():
        print(f"Could not open {filepath}")
        return []

    tree = file.Get("events")
    if not tree:
        print(f"No 'events' tree in {filepath}")
        return []

    accepted_ids = []

    for event in tree:
        eid = int(getattr(event, "eventid")[0])
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

        if not reasons:
            accepted_ids.append(eid)
        else:
            reason_tracker[method_label][eid].update(reasons)

    file.Close()
    return accepted_ids

# Convert reason dict to DataFrame
def reason_dict_to_df(reason_dict, method):
    data = []
    for eid, reasons in reason_dict.items():
        row = {"event_id": eid, "method": method}
        for reason in filter_names:
            row[reason] = reason in reasons
        data.append(row)
    return pd.DataFrame(data)

# Main loop
reason_tracker = {
    "GMM": defaultdict(set),
    "RANSAC": defaultdict(set)
}

for energy in excitation_energies:
    for cm in cm_angles:
        pattern = os.path.join(base_path, f"final_sim_5000_{energy}mev_{cm}cm_*_*_1.root")
        file_list = glob.glob(pattern)

        for filepath in file_list:
            process_file(filepath, "ransac_angles", "ransac_phi_angles",
                         "ransac_start", "ransac_end", "ransac_inter", "Elab", "RANSAC", reason_tracker)

            process_file(filepath, "gmm_angles", "gmm_phi_angles",
                         "gmm_start", "gmm_end", "gmm_inter", "Elab", "GMM", reason_tracker)

# Create combined DataFrame
df_gmm = reason_dict_to_df(reason_tracker["GMM"], "GMM")
df_ransac = reason_dict_to_df(reason_tracker["RANSAC"], "RANSAC")
df_all = pd.concat([df_gmm, df_ransac], ignore_index=True)

# Export to CSV (optional)
df_all.to_csv("filter_rejections_gmm_ransac_5cm_e1e2metricno.csv", index=False)

