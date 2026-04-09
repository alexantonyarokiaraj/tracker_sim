import ROOT
import glob
import numpy as np
import os
import sys
import pandas as pd
from scipy import interpolate
from collections import defaultdict


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from libraries import VolumeBoundaries, RunParameters


range_lookup_table = RunParameters.range_lookup_table.value
range_energy_conversion_sheet = RunParameters.range_energy_conversion_sheet.value
excel_data_df = pd.read_excel(range_lookup_table, sheet_name=range_energy_conversion_sheet)
table_range_energy = np.array(excel_data_df[['Energy(keV)', 'Range(mm)']])


def range_energy_calculate(range_value, reverse=False):
        if reverse:
            energy_table = table_range_energy[:, 1]
            range_table = table_range_energy[:, 0]
            table_low_energy = 3.39  # lowest possible range in mm
            table_high_energy = 51470  # highest possible range in mm
            table_low_range = 10  # lowest possible energy in kev
            table_high_range = 70000  # highest possible energy in kev
        else:
            energy_table = table_range_energy[:, 0]
            range_table = table_range_energy[:, 1]
            table_low_range = 3.39  # lowest possible range in mm
            table_high_range = 51470  # highest possible range in mm
            table_low_energy = 10  # lowest possible energy in kev
            table_high_energy = 70000  # highest possible energy in kev
        if range_value < table_low_range:
            required_energy = table_low_energy
        else:
            if range_value > table_high_range:
                required_energy = table_high_energy
            else:
                idx = (np.abs(range_table - range_value)).argmin()
                low_range = range_table[idx]
                low_energy = energy_table[idx]

                if low_range <= range_value:
                    high_range = range_table[idx + 1]
                    high_energy = energy_table[idx + 1]
                else:
                    high_range = range_table[idx - 1]
                    high_energy = energy_table[idx - 1]
                # Ensure low_range is the lesser and high_range is the greater
                if low_range > high_range:
                    low_range, high_range = high_range, low_range
                    low_energy, high_energy = high_energy, low_energy
                f = interpolate.interp1d([low_range, high_range], [low_energy, high_energy])
                required_energy = f(range_value)
                # print(type(required_energy), required_energy)
        return np.round(required_energy, 2)

def is_inside_volume(vertex_x, vertex_y, vertex_z):
    """
    Checks if the given vertex is inside the defined volume boundaries.

    Parameters:
    vertex_x (float): X coordinate of the vertex.
    vertex_y (float): Y coordinate of the vertex.
    vertex_z (float): Z coordinate of the vertex.

    Returns:
    bool: True if the vertex is inside the volume, False otherwise.
    """
    return (VolumeBoundaries.VOLUME_MIN.value <= vertex_x <= VolumeBoundaries.VOLUME_MAX.value and
            VolumeBoundaries.VOLUME_MIN.value <= vertex_y <= VolumeBoundaries.VOLUME_MAX.value and
            VolumeBoundaries.VOLUME_MIN.value <= vertex_z <= VolumeBoundaries.VOLUME_MAX.value)


# Define excitation energy and CM values
excitation_energies = [10]
cm_values = [1,2,3,4,5]


# Dictionary to store graphs for each (ex, cm) combination
graphs = {}

# Loop through excitation energy and CM combinations
for ex in excitation_energies:
    for cm in cm_values:
        file_path = "/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/alpha/"
        file_pattern = f"alpha_sim_5000_{ex}mev_{cm}cm_*.root"
        file_list = glob.glob(file_path + file_pattern)

        # Dictionary to store histograms for unique alpha_values in this (ex, cm) combination
        alpha_histograms = {}
        energy = []
        en_range = []
        # Create a TChain for the "events" tree
        chain = ROOT.TChain("events")

        # Add all files to the TChain
        for filename in file_list:
            chain.Add(filename)

        # Create histograms and process data
        for entry in range(chain.GetEntries()):
            chain.GetEntry(entry)
            if chain.eventid[0] > 0:
                labels = chain.gmm_ranges_alpha_labels
                counts = chain.gmm_ranges_alpha_counts
                alpha_values = chain.gmm_ranges_alpha
                initial_values = chain.gmm_ranges_initial
                phi_angles = chain.gmm_phi_angles
                vertex_values = chain.gmm_inter
                gmm_angles = np.array(chain.gmm_angles)
                gmm_input_range = range_energy_calculate(chain.Eenergy[0] * 1000, reverse=True)
                energy.append(chain.Eenergy[0] * 1000)
                en_range.append(gmm_input_range)

                if gmm_angles.size == 0:
                    continue

                closest_index = int(np.argmin(np.abs(gmm_angles - chain.Elab[0])))

                track_start = 0
                vertex_start = 0
                for track_idx in range(labels.size()):
                    label = labels[track_idx]
                    count = counts[track_idx]

                    if track_idx != closest_index:
                        continue

                    # Check phi_angle exclusion range
                    if np.any((70 <= phi_angles[track_idx]) & (phi_angles[track_idx] <= 110)) or \
                        np.any((-110 <= phi_angles[track_idx]) & (phi_angles[track_idx] <= -70)):
                        continue  # Skip if within exclusion range

                    vertex_x = vertex_values[vertex_start]
                    vertex_y = vertex_values[vertex_start+1]
                    vertex_z = vertex_values[vertex_start+2]

                    if not is_inside_volume(vertex_x, vertex_y, vertex_z):
                        continue

                    for i in range(track_start, track_start + count):
                        alpha = int(round(alpha_values[i], 4)*1e4)
                        diff = initial_values[i] - gmm_input_range

                        if alpha not in alpha_histograms:
                            alpha_histograms[alpha] = ROOT.TH1F(f"hist_{ex}_{cm}_{alpha}", f"Alpha: {alpha}", 100, -50, 50)
                            # print(alpha_histograms[alpha], alpha, chain.eventid[0])
                            ROOT.SetOwnership(alpha_histograms[alpha], True)
                        try:
                            alpha_histograms[alpha].Fill(diff)
                        except:
                            # print(alpha, ex,cm, alpha_histograms[alpha], alpha, chain.eventid[0])
                            pass

                    track_start += count
                    vertex_start += 3



        # Compute Mean for Each Alpha Value
        alpha_means = []
        alpha_keys = []

        for alpha, hist in alpha_histograms.items():
            mean_value = hist.GetMean()  # Compute mean of the histogram
            alpha_means.append(mean_value)
            alpha_keys.append(alpha/10000)

        # Convert to NumPy arrays for TGraph
        alpha_keys = np.array(alpha_keys, dtype=np.float64)
        alpha_means = np.array(alpha_means, dtype=np.float64)

        indices_closest_zero = np.where(np.abs(alpha_means) == np.min(np.abs(alpha_means)))[0] # Get indices of all closest values
        alpha_keys_at_closest_zero = alpha_keys[indices_closest_zero]

        if len(alpha_keys_at_closest_zero) > 1:
            print("Multiple alpha_keys values are equally close to zero:")
            for key in alpha_keys_at_closest_zero:
                print(key, energy[0], en_range[0])
        else:
            print(f"The alpha_keys value where alpha_means is closest to zero is: {alpha_keys_at_closest_zero[0]}, {energy[0]}, {en_range[0]}")

        # Create TGraph for (ex, cm)
        graph = ROOT.TGraph(len(alpha_keys), alpha_keys, alpha_means)
        graph.SetTitle(f"Mean (Initial - gmm_input_range) vs Alpha ({ex} MeV, {cm} cm);Alpha;Mean")
        graph.SetMarkerStyle(20)
        graph.SetMarkerColor(ROOT.kBlue)
        graph.SetLineColor(ROOT.kRed)

        # Store graph
        graphs[(ex, cm)] = graph

# Draw all graphs on a single canvas
canvas = ROOT.TCanvas("canvas", "Mean vs Alpha for All (Ex, CM) Combinations", 800, 600)
legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)

colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kCyan, ROOT.kOrange]
color_index = 0

first_graph = True
for (ex, cm), graph in graphs.items():
    graph.SetMarkerColor(colors[color_index % len(colors)])
    graph.SetLineColor(colors[color_index % len(colors)])
    legend.AddEntry(graph, f"{ex} MeV, {cm} cm", "lp")

    if first_graph:
        graph.Draw("AP")
        first_graph = False
    else:
        graph.Draw("P SAME")

    color_index += 1

legend.Draw()
canvas.Update()
canvas.WaitPrimitive()