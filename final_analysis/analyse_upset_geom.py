import ROOT
import glob
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from scipy import interpolate
import matplotlib.pyplot as plt
# from upsetplot import from_indicators, plot as upset_plot
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from libraries import FileNames
import math

# ==== Settings ====
excitation_energies = [10]
cm_angles = [4]
base_path = "/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/final/gamma_2_run/"
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

# ==== Beam Vector for Phi Calculation ====
beam_start = np.array([0, 128, 128])
beam_end = np.array([256, 128, 128])
beam_vector = beam_end - beam_start
beam_vector = beam_vector / np.linalg.norm(beam_vector)

# ==== Interpolation Table (mock sample) ====
# Replace with actual _TABLE_REVERSED from your experiment
range_lookup_table = FileNames.CONFIG_FILE_EXCEL.value
sheet_ = FileNames.RANGE_ENERGY_CONVERSION_SHEET.value
excel_data_df = pd.read_excel(range_lookup_table, sheet_name=sheet_)
_TABLE = np.array(excel_data_df[['Range(mm)', 'Energy(keV)']])
_TABLE = _TABLE[_TABLE[:, 0].argsort()]
_TABLE_REVERSED = _TABLE[:, [1, 0]]  # Now it's [Energy, Range]
# Optional: sort by energy just in case
_TABLE_REVERSED = _TABLE_REVERSED[_TABLE_REVERSED[:, 0].argsort()]

def truncate(number, digits):
    factor = 10.0 ** digits
    return math.trunc(number * factor) / factor

# ==== Helper Functions ====
def is_inside_volume(point):
    return all(volume_min <= coord <= volume_max for coord in point)

def is_outside_beam_zone(y):
    return not (beam_zone_min <= y <= beam_zone_max)

def extract_1track_vector(vec):
    return [vec[0], vec[1], vec[2]]

def calculate_phi_angle(v, beam_v):
    """
    Calculate the angle (in degrees) of a single vector in the YZ plane
    relative to the positive Y-axis.

    Parameters:
        v (array-like): The vector (3D).

    Returns:
        float: The signed angle of the vector in the YZ plane in degrees.
               Returns 400 if the vector has no magnitude in the YZ plane.
    """
    # Project the vector onto the YZ plane (ignore x-component)
    v_yz = np.array([v[1], v[2]])

    # Compute the norm of the projected vector
    norm_v = np.linalg.norm(v_yz)

    # Handle zero-magnitude vector
    if norm_v == 0:
        return 400  # Return 400 for zero magnitude in YZ plane

    # Reference direction is along the positive Y-axis
    ref_vector = np.array([1, 0])  # Positive Y-axis in YZ plane

    # Compute the dot product and angle
    dot_product = np.dot(v_yz, ref_vector)
    cos_theta = dot_product / norm_v
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure valid range for arccos
    angle = np.degrees(np.arccos(cos_theta))

    # Compute the cross product to determine the sign of the angle
    cross_product_z = v_yz[0] * ref_vector[1] - v_yz[1] * ref_vector[0]

    # Determine the sign of the angle
    if cross_product_z < 0:
        angle = -angle  # Negative direction

    return angle

def energy_range_calculate(energy_value):
    energy_table = _TABLE_REVERSED[:, 0]
    range_table = _TABLE_REVERSED[:, 1]

    # Handle out-of-bound energy values
    if energy_value <= energy_table[0]:
        return truncate(range_table[0], 2)
    elif energy_value >= energy_table[-1]:
        return truncate(range_table[-1], 2)

    # Interpolate range from energy
    f = interpolate.interp1d(energy_table, range_table)
    required_range = f(energy_value)

    return truncate(required_range, 2)

# ==== Main Processing for Reconstructed Methods ====
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

# ==== Processing for Geometric Truth ====
def process_geometric(filepath, reason_tracker):
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

        ver = np.array([event.verX[0], event.verY[0], event.verZ[0]])
        direction = np.array([event.dirX[0], event.dirY[0], event.dirZ[0]])
        energy = event.Eenergy[0] * 1000  # convert MeV to keV if needed

        range_ = energy_range_calculate(energy)
        end = ver + direction * range_

        phi_angle = calculate_phi_angle(direction, beam_vector)

        if 70 <= abs(phi_angle) <= 110:
            reasons.append("phi_cut")

        if not is_inside_volume(end):
            reasons.append("end_outside")

        if not is_inside_volume(ver):  # ver replaces inter
            reasons.append("inter_outside")

        if not is_outside_beam_zone(end[1]):
            reasons.append("end_in_beam_zone")

        if not reasons:
            accepted_ids.append(eid)
        else:
            reason_tracker["Geometric"][eid].update(reasons)

    file.Close()
    return accepted_ids

# ==== Convert to DataFrame ====
def reason_dict_to_df(reason_dict, method):
    data = []
    for eid, reasons in reason_dict.items():
        row = {"event_id": eid, "method": method}
        for reason in filter_names:
            row[reason] = reason in reasons
        data.append(row)
    return pd.DataFrame(data)

# ==== Run All ====
reason_tracker = {
    "GMM": defaultdict(set),
    "RANSAC": defaultdict(set),
    "Geometric": defaultdict(set)
}

for energy in excitation_energies:
    for cm in cm_angles:
        pattern = os.path.join(base_path, f"gamma_sim_5000_{energy}mev_{cm}cm_*_*.root")
        file_list = glob.glob(pattern)

        for filepath in file_list:
            process_file(filepath, "ransac_angles", "ransac_phi_angles",
                         "ransac_start", "ransac_end", "ransac_inter", "Elab", "RANSAC", reason_tracker)

            process_file(filepath, "gmm_angles", "gmm_phi_angles",
                         "gmm_start", "gmm_end", "gmm_inter", "Elab", "GMM", reason_tracker)

            process_geometric(filepath, reason_tracker)


# ==== Combine Data ====
df_gmm = reason_dict_to_df(reason_tracker["GMM"], "GMM")
df_ransac = reason_dict_to_df(reason_tracker["RANSAC"], "RANSAC")
df_geom = reason_dict_to_df(reason_tracker["Geometric"], "Geometric")
df_all = pd.concat([df_gmm, df_ransac, df_geom], ignore_index=True)

df_all.to_csv("filter_rejections_gmm_ransac_geometric_4cm.csv", index=False)

# ==== Optional: Plotting ====
# def plot_upset(df, method_label):
#     df_method = df[df["method"] == method_label]
#     upset_data = from_indicators(filter_names, df_method)
#     plt.figure(figsize=(10, 6))
#     upset_plot(upset_data, sort_by='cardinality')
#     plt.title(f"UpSet Plot: {method_label}")
#     plt.tight_layout()
#     plt.savefig(f"upset_plot_{method_label}.png")
#     plt.show()

# for method in ["GMM", "RANSAC", "Geometric"]:
#     plot_upset(df_all, method)
