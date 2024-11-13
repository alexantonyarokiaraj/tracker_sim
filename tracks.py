#######################################
# Final Code to populate simulated data
#######################################

#######################################
# Imports
#######################################

import sys
from ROOT import TFile, TH1F, TSpectrum, TF1, TH2F
import ROOT as root
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Button
import matplotlib.patches as patches
from matplotlib import cm, colors
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import namedtuple
import math
from ransac import find_multiple_lines_ransac
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from itertools import chain
from merger import calculate_cluster_metrics
import json
import os
from sklearn.metrics import adjusted_rand_score
from write import create_tree_and_branches, fill_event_data_to_tree
import pickle

np.random.seed(42)

#######################################
# Input arguments
#######################################

# excitation_energies = range(0, 31, 5)
# cm_angles = ["0_2", "0_3", "0_5", "1", "1_2", "1_5", "2", "3", "4", "5", "6", "7"]
arguments = sys.argv
input_string = arguments[1]
split_strings = input_string.split('@')
excitation_energies=[split_strings[0]]
cm_angles=[split_strings[1]]
path = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/DATA/SIMULATION/5000/"
plots = True
sim = True
debug=False
final_plots_flag = False
event_start =320
event_end = 323
save_final_data=False
with_missing_pads = True
batch_mode = False
save_to_root = False
save_python_figures = False

np.set_printoptions(threshold=np.inf)

if batch_mode:
    os.environ["DISPLAY"] = ""
    root.gROOT.SetBatch(True)
    # Check for display environment (for headless mode on servers, for example)
    if os.getenv("DISPLAY") is None:
        plt.switch_backend('Agg')  # Use a non-interactive backend to silence display

#######################################
# Configuration
#######################################

NB_COBO = 16
NB_ASAD = 4
NB_AGET = 4
NB_CHANNEL = 68
beam_entrance_time = 8.96  # units [ns]
beam_center_time = 9980.0 - beam_entrance_time  # units[ns], 7076.0
beam_center_peak_find_low = 8000  # units[ns]
beam_center_peak_find_high = 14000  # units[ns]
sig_beam_center = 70.0  # units [ns]
time_per_sample = 0.08  # units [us]
drift_velocity_volume = 1.28  # units [cm/us]
table = np.loadtxt("LT_GANIL_NewCF_marine.dat")
z_conversion_factor = drift_velocity_volume*(10.0/1000.0)
x_conversion_factor = 2.0  # units[mm]
y_conversion_factor = 2.0  # units[mm]
missing_pads_info = "HitResponses.dat"
missed_pads = np.loadtxt(missing_pads_info)
x_pos_raw = missed_pads[:,0]
y_pos_raw = missed_pads[:,1]
nbins_x = 128
x_start_bin = 0 * x_conversion_factor
x_end_bin = 128 * x_conversion_factor
nbins_y = 128
y_start_bin = 0 * y_conversion_factor
y_end_bin = 128 * y_conversion_factor
nbins_z = 28000
z_start_bin = 0 * z_conversion_factor
z_end_bin = 28000 * z_conversion_factor
pixel_size_mm = 2
line_length = 100
transparency = 0.7
beam_zone_low = 122
beam_zone_high = 132

#######################################
# Graphics
#######################################

fig, axs = plt.subplots(4, 4, figsize=(20, 10))

ax1 = axs[0,0]
ax2 = axs[0,1]
ax3 = axs[0,2]
ax4 = axs[0,3]
ax5 = axs[1,0]
ax6 = axs[1,1]
ax7 = axs[1,2]
ax8 = axs[2,0]
ax9 = axs[2,1]
ax10 = axs[2,2]
ax11 = axs[3,0]
ax12 = axs[3,1]
ax13 = axs[3,2]


# Function to add rectangle patches for each point
def add_rectangles(ax, xyz_data, labels, cmap, proj, colorbarFlag, discrete=False):
    unique_labels = np.unique(labels)
    if discrete:
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        mapped_labels = np.array([label_mapping[label] for label in labels])
        labels = mapped_labels
    for (x, y, z), label in zip(xyz_data, labels):
        color = cmap(int(label) % cmap.N)
        if proj == 'yz':
            x = y
            y = z
        if proj == 'xz':
            x = x
            y = z
        rect = patches.Rectangle((x, y), pixel_size_mm, pixel_size_mm, linewidth=0.5,
                                 edgecolor='none', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
    if colorbarFlag:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if fig.get_axes().count(cax):  # Check if cax exists in figure axes
            cax.cla()  # Clear the colorbar axis to reset it
        norm = colors.Normalize(vmin=0, vmax=len(unique_labels) - 1)  # Update normalization
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Avoid warnings
        colorbar = fig.colorbar(sm, cax=cax, orientation='vertical', label='Intensity')
        if discrete:
            colorbar.set_ticks(np.arange(len(unique_labels)))
            colorbar.set_ticklabels(unique_labels)  # Set original label values
        return colorbar
    else:
        return None

# Function to set custom grid and sparse Y-axis tick labels
def set_custom_grid(ax):
    x_limits = (0, 256)
    y_limits = (0, 256)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    # Set grid ticks every 2 mm on both axes
    ax.set_xticks(np.arange(x_limits[0], x_limits[1] + 1, 20), minor=False)
    ax.set_yticks(np.arange(y_limits[0], y_limits[1] + 1, pixel_size_mm), minor=True)
    ax.set_xticks(np.arange(x_limits[0], x_limits[1] + 1, pixel_size_mm), minor=True)
    ax.set_yticks(np.arange(y_limits[0], y_limits[1] + 1, 20), minor=False)
    # Display grid
    ax.grid(which="both", color="lightgray", linestyle="--", linewidth=0.5)
    # Label Y-axis major ticks every 20 mm
    y_labels = np.arange(y_limits[0], y_limits[1] + 1, 20)
    ax.set_yticklabels(y_labels)
    # Label X-axis major ticks every 20 mm
    x_labels = np.arange(x_limits[0], x_limits[1] + 1, 20)
    ax.set_xticklabels(x_labels)

# Function to clear and update axes
def update_clear(ax):
    ax.clear()
    if ax in [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13]:
        set_custom_grid(ax)
        if ax in  [ax2, ax5, ax8, ax11]:
            xlabel = 'X [mm]'
            ylabel =  'Y [mm]'
        if ax in [ax3, ax6, ax9, ax12]:
            xlabel = 'Y [mm]'
            ylabel =  'Z [mm]'
        if ax in [ax4, ax7, ax10, ax13]:
            xlabel = 'X [mm]'
            ylabel =  'Z [mm]'
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    if ax in [ax1]:
        ax.set_xlabel('Z', fontsize=12, fontweight='bold')
        ax.set_ylabel('Counts', fontsize=12, fontweight='bold')

update_clear(ax1)
update_clear(ax2)
update_clear(ax3)
update_clear(ax4)
update_clear(ax5)
update_clear(ax6)
update_clear(ax7)
update_clear(ax8)
update_clear(ax9)
update_clear(ax10)
update_clear(ax11)
update_clear(ax12)
update_clear(ax13)

next_pressed = False
# Create a button for going to the next entry
ax_next = plt.axes([0.45, 0.01, 0.1, 0.05])  # Button position and size
button_next = Button(ax_next, 'Next')

# Callback function to handle button clicks
def next_button_callback(event):
    global next_pressed
    next_pressed = True  # Set the flag to True when the button is pressed

# Attach the callback to the button
button_next.on_clicked(next_button_callback)

#######################################
# Supporting Functions
#######################################
# Function to calculate the angle between two vectors
def angle_between(v1, v2):
    # Ensure the direction of v1 aligns with v2 by flipping if necessary
    # if np.dot(v1, v2) < 0:
    #     v1 = -v1  # Flip v1 to ensure it points in the same general direction as v2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

#Function to find the peak in Z spectrum
def get_gpeaks(h, lrange, sigma, opt, thres, niter):
        s = TSpectrum(niter)
        h.GetXaxis().SetRange(lrange[0], lrange[1])
        try:
            s.Search(h, sigma, opt, thres)
            bufX, bufY = s.GetPositionX(), s.GetPositionY()
            pos = []
            for i in range(s.GetNPeaks()):
                pos.append([bufX[i], bufY[i]])
            pos.sort()
            return pos
        except:
            return [[beam_center_time, 10]]

#Function to get beam entrance time
def get_beam_center(entries):
    z_proj = TH1F('z_proj', 'z_proj', 20000, 0, 20000)
    z_proj.Reset()
    length = entries.data.CoboAsad.size()
    z_proj_arr = []
    for xq in range(length):
        co = entries.data.CoboAsad[xq].globalchannelid >> 11
        if co != 31 and co != 16:
            asad = (entries.data.CoboAsad[xq].globalchannelid - (co << 11)) >> 9
            ag = (entries.data.CoboAsad[xq].globalchannelid - (co << 11) - (asad << 9)) >> 7
            ch = entries.data.CoboAsad[xq].globalchannelid - (co << 11) - (asad << 9) - (ag << 7)
            where = co * NB_ASAD * NB_AGET * NB_CHANNEL + asad * NB_AGET * NB_CHANNEL + ag * NB_CHANNEL + ch
            hitlength = entries.data.CoboAsad[int(xq)].peakheight.size()
            for yq in range(0, hitlength):
                posZ = entries.data.CoboAsad[int(xq)].peaktime[int(yq)]
                if not entries.data.CoboAsad[xq].hasSaturation:
                    z_proj.Fill(entries.data.CoboAsad[int(xq)].peaktime[int(yq)])
                    z_proj_arr.append([entries.data.CoboAsad[int(xq)].peaktime[int(yq)]])
    peaks = get_gpeaks(z_proj,[beam_center_peak_find_low, beam_center_peak_find_high], 2, "",0.5, 10)
    xpeaks = np.array(peaks)
    try:
        if len(xpeaks[:, 0]) > 1:
            max_index = np.argmax(xpeaks[:, 1])
            max_location = xpeaks[max_index, 0]
            fit_gaus = TF1("fit_gaus", "gaus", max_location - 1000, max_location + 1000)
            z_proj.Fit(fit_gaus, "RQN0", "NOM", max_location - 4 * sig_beam_center,
                    max_location + 4 * sig_beam_center)
            beam_c = fit_gaus.GetParameter(1) - beam_entrance_time
        else:
            beam_c = beam_center_time - beam_entrance_time
    except:
        beam_c = beam_center_time - beam_entrance_time
    if plots:
        z_proj_nparr = np.array(z_proj_arr)
        update_clear(ax1)
        ax1.hist(z_proj_nparr, bins=50, range=(beam_center_time-1000,beam_center_time+1000), color='skyblue', alpha=0.7, label="Time")
        ax1.text(0.95, 0.95, f'Beam Center: {beam_center_time}',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax1.transAxes,  # Use axes coordinates
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5))
        plt.draw()
    return beam_c

# Function to read the input file into the array
def get_data_array(beam_center, entries, event_info):
    global axs
    data_points = []
    # print("Getting Data Array")
    length = entries.data.CoboAsad.size()
    for x in range(length):
        co = entries.data.CoboAsad[x].globalchannelid >> 11
        if co != 31 and co != 16:
            asad = (entries.data.CoboAsad[x].globalchannelid - (co << 11)) >> 9
            ag = (entries.data.CoboAsad[x].globalchannelid - (co << 11) - (asad << 9)) >> 7
            ch = entries.data.CoboAsad[x].globalchannelid - (co << 11) - (asad << 9) - (ag << 7)
            where = co * NB_ASAD * NB_AGET * NB_CHANNEL + asad * NB_AGET * NB_CHANNEL + ag * NB_CHANNEL + ch
            posX = (table[where][4] * x_conversion_factor)
            posY = (table[where][5] * y_conversion_factor)
            missing_pad_check = False
            posX_raw = table[where][4]
            posY_raw = table[where][5]
            t = missed_pads[(missed_pads[:, 0] == posX_raw) & (missed_pads[:, 1] == posY_raw)]
            if len(t) >= 1:
                if with_missing_pads:
                    missing_pad_check = True
                else:
                    missing_pad_check = False
            hitlength = entries.data.CoboAsad[int(x)].peakheight.size()
            for y in range(0, hitlength):
                if entries.data.input_ejectile_mom_dirY >=0:
                    posZ = (beam_center + (beam_center - entries.data.CoboAsad[int(x)].peaktime[int(y)])) * z_conversion_factor
                else:
                    posZ = (beam_center - (entries.data.CoboAsad[int(x)].peaktime[int(y)]-beam_center)) * z_conversion_factor
                posZ = entries.data.CoboAsad[int(x)].peaktime[int(y)]* z_conversion_factor
                Qvox = entries.data.CoboAsad[int(x)].peakheight[int(y)]
                if not entries.data.CoboAsad[x].hasSaturation and not missing_pad_check:
                    data_points.append([posX, posY, posZ, Qvox])
    data = np.array(data_points)
    if plots:
        charge = data[:,3]
        cmap = plt.cm.get_cmap("viridis", len(charge))
        update_clear(ax2)
        update_clear(ax3)
        update_clear(ax4)
        colorbar1 = add_rectangles(ax2, data[:, 0:3], charge, cmap, proj = 'xy', colorbarFlag=True)
        colorbar2 = add_rectangles(ax3, data[:, 0:3], charge, cmap, proj = 'yz', colorbarFlag=True)
        colorbar3 = add_rectangles(ax4, data[:, 0:3], charge, cmap, proj = 'xz', colorbarFlag=True)
        num_data_points = len(data[:,0])
        ax2.plot([event_info.verX, event_info.verX + line_length * event_info.dirX],[event_info.verY, event_info.verY + line_length * event_info.dirY],
                 color='blue', alpha=transparency)
        ax2.scatter(event_info.verX, event_info.verY, color='red', edgecolor='black', s=50, zorder=3)  # Circle for vertex
        # Place the number of data points in the top right corner
        ax2.text(0.95, 0.95, f'Count: {num_data_points}, E: {event_info.Eenergy}, L: {event_info.Elab}',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax2.transAxes,  # Use axes coordinates
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))
        ax3.plot([event_info.verY, event_info.verY + line_length * event_info.dirY],[event_info.verZ, event_info.verZ + line_length * event_info.dirZ],
                 color='blue', alpha=transparency)
        ax3.scatter(event_info.verY, event_info.verZ, color='red', edgecolor='black', s=50, zorder=3)  # Circle for vertex
        ax3.text(0.95, 0.95, f'Count: {num_data_points}, E: {event_info.Eenergy}, L: {event_info.Elab}',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax3.transAxes,  # Use axes coordinates
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))
        ax4.plot([event_info.verX, event_info.verX + line_length * event_info.dirX],[event_info.verZ, event_info.verZ + line_length * event_info.dirZ],
                 color='blue', alpha=transparency)
        ax4.scatter(event_info.verX, event_info.verZ, color='red', edgecolor='black', s=50, zorder=3)  # Circle for vertex
        ax4.text(0.95, 0.95, f'Count: {num_data_points}, E: {event_info.Eenergy}, L: {event_info.Elab}',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax4.transAxes,  # Use axes coordinates
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))
        plt.draw()
        return data, [colorbar1, colorbar2, colorbar3]
    # print('Checking plot return')
    return data

# Function to plot true labels
def generate_true_labels(data_array, event_info):
    y_values = data_array[:,1]
    labels = np.where((y_values >= beam_zone_low) & (y_values < beam_zone_high), 0, 1)
    data_array = np.column_stack((data_array, labels))
    if plots:
        cmap = plt.cm.get_cmap("Dark2", len(np.unique(labels)))
        update_clear(ax5)
        update_clear(ax6)
        update_clear(ax7)
        colorbar4 = add_rectangles(ax5, data_array[:, 0:3], labels, cmap, proj = 'xy', colorbarFlag=False, discrete=True)
        colorbar5 = add_rectangles(ax6, data_array[:, 0:3], labels, cmap, proj = 'yz', colorbarFlag=False, discrete=True)
        colorbar6 = add_rectangles(ax7, data_array[:, 0:3], labels, cmap, proj = 'xz', colorbarFlag=False, discrete=True)
        num_data_points = len(data_array[:,0])
        # Place the number of data points in the top right corner
        ax5.plot([event_info.verX, event_info.verX + line_length * event_info.dirX],[event_info.verY, event_info.verY + line_length * event_info.dirY],
                 color='blue', alpha=transparency)
        ax5.scatter(event_info.verX, event_info.verY, color='red', edgecolor='black', s=50, zorder=3)  # Circle for vertex
        ax5.text(0.95, 0.95, f'Count: {num_data_points}, E: {event_info.Eenergy}, L: {event_info.Elab}',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax5.transAxes,  # Use axes coordinates
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))
        ax6.plot([event_info.verY, event_info.verY + line_length * event_info.dirY],[event_info.verZ, event_info.verZ + line_length * event_info.dirZ],
                 color='blue', alpha=transparency)
        ax6.scatter(event_info.verY, event_info.verZ, color='red', edgecolor='black', s=50, zorder=3)  # Circle for vertex
        ax6.text(0.95, 0.95, f'Count: {num_data_points}, E: {event_info.Eenergy}, L: {event_info.Elab}',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax6.transAxes,  # Use axes coordinates
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))
        ax7.plot([event_info.verX, event_info.verX + line_length * event_info.dirX],[event_info.verZ, event_info.verZ + line_length * event_info.dirZ],
                 color='blue', alpha=transparency)
        ax7.scatter(event_info.verX, event_info.verZ, color='red', edgecolor='black', s=50, zorder=3)  # Circle for vertex
        ax7.text(0.95, 0.95, f'Count: {num_data_points}, E: {event_info.Eenergy}, L: {event_info.Elab}',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax7.transAxes,  # Use axes coordinates
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))
        plt.draw()
    return data_array

# Function to plot clusters from ransac
def plot_ransac(data_array, event_info):
    labels = data_array[:, 5]
    cmap = plt.cm.get_cmap("Dark2", len(np.unique(labels)))
    update_clear(ax8)
    update_clear(ax9)
    update_clear(ax10)
    colorbar4 = add_rectangles(ax8, data_array[:, 0:3], labels, cmap, proj = 'xy', colorbarFlag=True, discrete=True)
    colorbar5 = add_rectangles(ax9, data_array[:, 0:3], labels, cmap, proj = 'yz', colorbarFlag=True, discrete=True)
    colorbar6 = add_rectangles(ax10, data_array[:, 0:3], labels, cmap, proj = 'xz', colorbarFlag=True, discrete=True)
     # Place the number of data points in the top right corner
    ax8.plot([event_info.verX, event_info.verX + line_length * event_info.dirX],[event_info.verY, event_info.verY + line_length * event_info.dirY],
                color='blue', alpha=transparency)
    ax8.scatter(event_info.verX, event_info.verY, color='red', edgecolor='black', s=50, zorder=3)  # Circle for vertex
    ax9.plot([event_info.verY, event_info.verY + line_length * event_info.dirY],[event_info.verZ, event_info.verZ + line_length * event_info.dirZ],
                 color='blue', alpha=transparency)
    ax9.scatter(event_info.verY, event_info.verZ, color='red', edgecolor='black', s=50, zorder=3)  # Circle for vertex
    ax10.plot([event_info.verX, event_info.verX + line_length * event_info.dirX],[event_info.verZ, event_info.verZ + line_length * event_info.dirZ],
                 color='blue', alpha=transparency)
    ax10.scatter(event_info.verX, event_info.verZ, color='red', edgecolor='black', s=50, zorder=3)  # Circle for vertex
    plt.draw()
    return [colorbar4, colorbar5, colorbar6]

# Function to plot the threshold distance in RANSAC
def plot_threshold_lines(start_point, direction_vector):
    # Normalize the direction vector
    direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)

    # Threshold distance (5 mm) perpendicular to the line
    threshold_distance = 5

    # Generate points along the fitted line for plotting
    t_values = np.linspace(-50, 50, 100)  # Adjust range as needed
    fitted_line_points = start_point + t_values[:, None] * direction_vector_normalized

    # Create perpendicular vectors in the plane perpendicular to the direction vector
    # Find any vector orthogonal to the direction vector
    arbitrary_vector = np.array([1, 0, 0]) if direction_vector_normalized[0] == 0 else np.array([0, 1, 0])
    perpendicular_vector1 = np.cross(direction_vector_normalized, arbitrary_vector)
    perpendicular_vector1 /= np.linalg.norm(perpendicular_vector1)

    # Create another perpendicular vector orthogonal to both direction_vector and perpendicular_vector1
    perpendicular_vector2 = np.cross(direction_vector_normalized, perpendicular_vector1)
    perpendicular_vector2 /= np.linalg.norm(perpendicular_vector2)

    # Calculate points at threshold distances in both perpendicular directions
    threshold_line_above1 = fitted_line_points + threshold_distance * perpendicular_vector1
    threshold_line_below1 = fitted_line_points - threshold_distance * perpendicular_vector1
    threshold_line_above2 = fitted_line_points + threshold_distance * perpendicular_vector2
    threshold_line_below2 = fitted_line_points - threshold_distance * perpendicular_vector2


    return threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2

# Function to plot the kinematics of RANSAC Clusters
def kinematics_ransac(data_array, fitted_models, useLineModelND):
    # Define the beam line endpoints

    angles = {}
    low_energy_track_count = 0

    beam_line_start = np.array([0, 128, 128])
    beam_line_end = np.array([256, 128, 128])
    beam_line_direction = beam_line_end - beam_line_start
    beam_line_direction = beam_line_direction / np.linalg.norm(beam_line_direction)  # Normalize

    ransac_labels = data_array[:, 5]

    # Iterate through each unique label
    unique_labels = np.unique(ransac_labels[ransac_labels != 20])  # Exclude unassigned labels (20)
    for label in unique_labels:
        # Get the points corresponding to the current label
        cluster_points = data_array[ransac_labels == label]

        # Check if the cluster size is greater than 10
        if cluster_points.shape[0] <= 10:
            continue  # Skip this cluster if it has 10 or fewer points

        # Calculate the mean y of the cluster
        mean_y = np.mean(cluster_points[:, 1])

        # Process clusters either above or below the beam line
        if (mean_y >= beam_zone_high or mean_y < beam_zone_low) and cluster_points.shape[0] >= 10:
            # Print model parameters for debugging
            # print("Fitted model parameters for label", label, ":", fitted_models[label].params)
            low_energy_track_count += 1
            # Get the direction vector and start point from the fitted model
            model_params = fitted_models[label].params
            start_point = np.array(model_params[0])

            pca = PCA(n_components=1)
            pca.fit(cluster_points[:, :3])  # Only x, y, z coordinates

            if not useLineModelND:
                direction_vector = pca.components_[0]
                threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2 = plot_threshold_lines(start_point, direction_vector)
            if useLineModelND:
                direction_vector = np.array(model_params[1])
                threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2 = plot_threshold_lines(start_point, direction_vector)

            # Normalize the direction vector
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
            # Calculate angle with the beam line
            angle = angle_between(direction_vector, beam_line_direction)

            # Store results
            angles[label] = round(angle, 2)

            # Plot the line for this cluster in the XY projection
            # Extend the line for better visualization
            line_start = start_point - 50 * direction_vector
            line_end = start_point + 50 * direction_vector
            if plots:
                # Threshold line projections
                xy_threshold_above1 = threshold_line_above1[:, [0, 1]]
                yz_threshold_above1 = threshold_line_above1[:, [1, 2]]
                xz_threshold_above1 = threshold_line_above1[:, [0, 2]]

                xy_threshold_below1 = threshold_line_below1[:, [0, 1]]
                yz_threshold_below1 = threshold_line_below1[:, [1, 2]]
                xz_threshold_below1 = threshold_line_below1[:, [0, 2]]

                # Threshold line projections for the second perpendicular vector
                xy_threshold_above2 = threshold_line_above2[:, [0, 1]]
                yz_threshold_above2 = threshold_line_above2[:, [1, 2]]
                xz_threshold_above2 = threshold_line_above2[:, [0, 2]]

                xy_threshold_below2 = threshold_line_below2[:, [0, 1]]
                yz_threshold_below2 = threshold_line_below2[:, [1, 2]]
                xz_threshold_below2 = threshold_line_below2[:, [0, 2]]

                ax8.plot([line_start[0], line_end[0]],
                        [line_start[1], line_end[1]],
                        label=f"Cluster {label} (Angle: {angle:.2f}°)", linestyle="--")
                ax8.plot(xy_threshold_above1[:, 0], xy_threshold_above1[:, 1], 'r--', label='Threshold +5mm')
                ax8.plot(xy_threshold_below1[:, 0], xy_threshold_below1[:, 1], 'g--', label='Threshold -5mm')
                ax8.plot(xy_threshold_above2[:, 0], xy_threshold_above2[:, 1], 'r--')
                ax8.plot(xy_threshold_below2[:, 0], xy_threshold_below2[:, 1], 'g--')
                ax9.plot([line_start[1], line_end[1]],
                        [line_start[2], line_end[2]],
                        label=f"Cluster {label} (Angle: {angle:.2f}°)", linestyle="--")
                ax9.plot(yz_threshold_above1[:, 0], yz_threshold_above1[:, 1], 'r--', label='Threshold +5mm')
                ax9.plot(yz_threshold_below1[:, 0], yz_threshold_below1[:, 1], 'g--', label='Threshold -5mm')
                ax9.plot(yz_threshold_above2[:, 0], yz_threshold_above2[:, 1], 'r--')
                ax9.plot(yz_threshold_below2[:, 0], yz_threshold_below2[:, 1], 'g--')
                ax10.plot([line_start[0], line_end[0]],
                        [line_start[2], line_end[2]],
                        label=f"Cluster {label} (Angle: {angle:.2f}°)", linestyle="--")
                ax10.plot(xz_threshold_above1[:, 0], xz_threshold_above1[:, 1], 'r--', label='Threshold +5mm')
                ax10.plot(xz_threshold_below1[:, 0], xz_threshold_below1[:, 1], 'g--', label='Threshold -5mm')
                ax10.plot(xz_threshold_above2[:, 0], xz_threshold_above2[:, 1], 'r--')
                ax10.plot(xz_threshold_below2[:, 0], xz_threshold_below2[:, 1], 'g--')
                plt.draw
    return angles, low_energy_track_count

# Function to do the GMM Fitting
def fit_gmm_with_bic(data, max_components=10):
    """
    Fit Gaussian Mixture Model using BIC to select optimal components.

    Parameters:
    - data (np.ndarray): Input data array with shape (n_samples, 6) where columns are x, y, z, q, true labels, ransac labels, gmm labels.
    - max_components (int): Maximum number of GMM components to evaluate for BIC score.

    Returns:
    - best_labels (np.ndarray): Labels assigned to each data point for the GMM model with the lowest BIC.
    - best_n_components (int): Number of components in the best GMM model according to BIC.
    """
    # Extract features (first 3 columns: x, y, z, q)
    features = data[:, :3]

    best_bic = np.inf
    best_gmm = None
    best_n_components = 1

    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(features)
        bic = gmm.bic(features)

        # Check if this model has the lowest BIC
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n_components = n_components

    # Fit the best GMM model and predict labels
    best_labels = best_gmm.predict(features)

    return best_labels, best_n_components

# Function to plot the GMM assigned clusters
def plot_gmm(data_array, event_info):
    labels = data_array[:, 6]
    cmap = plt.cm.get_cmap("Dark2", len(np.unique(labels)))
    update_clear(ax11)
    update_clear(ax12)
    update_clear(ax13)
    colorbar7 = add_rectangles(ax11, data_array[:, 0:3], labels, cmap, proj = 'xy', colorbarFlag=True, discrete=True)
    colorbar8 = add_rectangles(ax12, data_array[:, 0:3], labels, cmap, proj = 'yz', colorbarFlag=True, discrete=True)
    colorbar9 = add_rectangles(ax13, data_array[:, 0:3], labels, cmap, proj = 'xz', colorbarFlag=True, discrete=True)
     # Place the number of data points in the top right corner
    ax11.plot([event_info.verX, event_info.verX + line_length * event_info.dirX],[event_info.verY, event_info.verY + line_length * event_info.dirY],
                color='blue', alpha=transparency)
    ax11.scatter(event_info.verX, event_info.verY, color='red', edgecolor='black', s=50, zorder=3)  # Circle for vertex
    ax12.plot([event_info.verY, event_info.verY + line_length * event_info.dirY],[event_info.verZ, event_info.verZ + line_length * event_info.dirZ],
                 color='blue', alpha=transparency)
    ax12.scatter(event_info.verY, event_info.verZ, color='red', edgecolor='black', s=50, zorder=3)  # Circle for vertex
    ax13.plot([event_info.verX, event_info.verX + line_length * event_info.dirX],[event_info.verZ, event_info.verZ + line_length * event_info.dirZ],
                 color='blue', alpha=transparency)
    ax13.scatter(event_info.verX, event_info.verZ, color='red', edgecolor='black', s=50, zorder=3)  # Circle for vertex
    plt.draw()
    # print('GMM colorbar', [colorbar7, colorbar8, colorbar9])
    return [colorbar7, colorbar8, colorbar9]

# Function to plot the kinematics of GMM Clusters
def kinematics_gmm(data):
    """
    Analyze clusters based on GMM labels to find direction vectors, intersections with a beam line,
    and angles. Plot the clusters with their fitted line in XY projection.

    Parameters:
    - data (np.ndarray): Input data array with columns [x, y, z, true labels, ransac labels, gmm labels].

    Returns:
    - intersections (dict): Dictionary of intersections for qualifying clusters.
    - angles (dict): Dictionary of angles (in degrees) between the cluster direction vectors and the beamline.
    """
    intersections = {}
    angles = {}

    # Group by GMM labels
    gmm_labels = np.unique(data[:, 6])

    low_energy_track_count = 0

    for label in gmm_labels:
        # Filter data for the current cluster
        cluster_data = data[data[:, 6] == label]
        if len(cluster_data) < 10:
            continue  # Skip clusters with fewer than 10 points

        # Calculate the mean of Y coordinate
        mean_y = np.mean(cluster_data[:, 1])
        if (mean_y >= beam_zone_high or mean_y < beam_zone_low) and cluster_data.shape[0] >= 10:


            # Calculate the direction vector
            end_point, start_point, beam_vector, dirVecTrackNorm, track_mean = get_directions(cluster_data[:, :3])
            track_vector = end_point - start_point

            # Calculate the angle between the cluster direction vector and the beamline
            angle = angle_between(track_vector, beam_vector)

            # Calculate the intersections
            intersection_point = closest_point_on_line1(start_point, track_vector, np.array([0,128,128]), beam_vector)

            # Store results
            angles[label] = round(angle, 2)
            intersections[label] = intersection_point

            # print(f"Label: {label}, Angle with beam line: {angle:.2f} degrees")

            # Plot the fitted line for the cluster in XY projection
            # Define two endpoints for the line segment along the direction vector
            line_length = 100  # Adjust line length as needed
            line_start = track_mean - line_length * dirVecTrackNorm
            line_end = track_mean + line_length * dirVecTrackNorm
            if plots:
                ax11.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], label=f'Fitted Line {label}')
                ax11.scatter(intersection_point[0], intersection_point[1], color='blue', marker='o', label='Intersection Point', s=100)
                ax12.plot([line_start[1], line_end[1]], [line_start[2], line_end[2]], label=f'Fitted Line {label}')
                ax12.scatter(intersection_point[1], intersection_point[2], color='blue', marker='o', label='Intersection Point', s=100)
                ax13.plot([line_start[0], line_end[0]], [line_start[2], line_end[2]], label=f'Fitted Line {label}')
                ax13.scatter(intersection_point[0], intersection_point[2], color='blue', marker='o', label='Intersection Point', s=100)
    return angles, low_energy_track_count

# Function to do final plots
def final_plots(axes_final, bin_range, bin_width, plot_list, notation):
    bins = np.arange(bin_range[0], bin_range[1] + bin_width, bin_width)
    axes_final.hist(plot_list, bins=bins, alpha=0.6, label=notation, histtype='step')
    axes_final.legend()

# Function to save final output
def save_list(energy, angle, list, arr_name):
    with open("output/no_missing_pads/"+energy+'mev_'+angle+"cm_"+arr_name+".json", "w") as file:
        json.dump(list, file)

# Function to calculate the number of unique beam and track gmm
def beam_track_data(data_array):
    # Ensure the data_array has at least 7 columns
    if data_array.shape[1] < 7:
        raise ValueError("data_array must have at least 7 columns")

    # Extract RANSAC and GMM labels
    ransac_labels = data_array[:, 5]
    gmm_labels = data_array[:, 6]
    y_values = data_array[:, 1]  # Assuming the second column is y

    # Create dictionaries to hold y-values for each RANSAC and GMM cluster
    ransac_clusters = {}
    gmm_clusters = {}

    for i in range(len(data_array)):
        ransac_label = ransac_labels[i]
        gmm_label = gmm_labels[i]
        y_value = y_values[i]

        # Group by RANSAC labels
        if ransac_label not in ransac_clusters:
            ransac_clusters[ransac_label] = []
        ransac_clusters[ransac_label].append(y_value)

        # Group by GMM labels
        if gmm_label not in gmm_clusters:
            gmm_clusters[gmm_label] = []
        gmm_clusters[gmm_label].append(y_value)

    # Classify clusters based on mean y values
    def classify_clusters(clusters):
        beam_count = 0
        track_count = 0
        unique_labels = set()

        for label, y_vals in clusters.items():
            mean_y = np.mean(y_vals)
            unique_labels.add(label)

            if 122 <= mean_y < 132:
                beam_count += 1  # Count as beam
            else:
                track_count += 1  # Count as track

        return beam_count, track_count, unique_labels

    # Classify RANSAC and GMM clusters
    unique_beam_ransac, unique_track_ransac, ransac_unique_labels = classify_clusters(ransac_clusters)
    unique_beam_gmm, unique_track_gmm, gmm_unique_labels = classify_clusters(gmm_clusters)

    # print(unique_beam_ransac, unique_track_ransac)
    # print(unique_beam_gmm, unique_track_gmm)
    return unique_beam_ransac, unique_track_ransac, unique_beam_gmm, unique_track_gmm

# Function to find closest points on line
def find_closest_points_on_line(data, direction_vector, cluster_mean):
    """
    Finds the closest points on a line passing through cluster_mean and oriented along direction_vector for each point in data.

    Args:
        data: A NumPy array of shape (n, 3) containing the track data.
        direction_vector: A NumPy array of shape (3,) representing the direction of the line.
        cluster_mean: A NumPy array of shape (3,) representing the point through which the line passes (mean of the data).

    Returns:
        A NumPy array of shape (n, 3) containing the closest points on the line for each point in data.
    """
    # Normalize the direction vector
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Translate points so cluster_mean is the origin
    centered_data = data - cluster_mean

    # Project each centered point onto the direction vector
    projections = np.dot(centered_data, direction_vector).reshape(-1, 1)

    # Get the closest points by moving along the direction vector and add the mean back
    closest_points = cluster_mean + projections * direction_vector

    return closest_points

# Function to find the start and the end points
def start_end_points(pca_points, beam_mean, dirVecBeam):
    """
    Finds the start and end points on the PCA line based on the shortest and longest distances
    to the beam line.

    Args:
        pca_points: A NumPy array of shape (n, 3) containing the closest points on the PCA line.
        beam_mean: A NumPy array of shape (3,) representing the starting point of the beam line.
        dirVecBeam: A NumPy array of shape (3,) representing the normalized direction vector of the beam line.

    Returns:
        start_point: The point on the PCA line with the shortest distance to the beam line.
        end_point: The point on the PCA line with the longest distance to the beam line.
        distances: A list of distances from each PCA point to the beam line.
    """
    distances = []
    closest_points_on_beam = []

    for point in pca_points:
        # Find the closest point on the beam line for each PCA point
        closest_point = find_closest_points_on_line(point, dirVecBeam, beam_mean)
        # Calculate the distance between the point on the PCA line and the closest point on the beam line
        distance = np.linalg.norm(point - closest_point)
        distances.append(distance)
        closest_points_on_beam.append(closest_point)

    # Convert distances to a numpy array for easier indexing
    distances = np.array(distances)

    # Find the index of the point with the smallest and largest distance
    start_index = np.argmin(distances)
    end_index = np.argmax(distances)

    # Return the start and end points based on the distances
    start_point = pca_points[start_index]
    end_point = pca_points[end_index]

    return start_point, end_point

# Function to plot the kinematics of GMM Clusters
def get_directions(data, beam_start=np.array([0, 128, 128]), beam_end=np.array([256, 128, 128])):
    pca = PCA(n_components=1)
    pca.fit(data)
    dirVecTrack = pca.components_[0]
    dirVecTrackNorm = dirVecTrack / np.linalg.norm(dirVecTrack)
    track_mean = pca.mean_
    closest_points = find_closest_points_on_line(data, dirVecTrack, track_mean)
    if plots:
        ax11.scatter(closest_points[:, 0], closest_points[:, 1], color='red', label='Closest Points on PCA Line')
        ax12.scatter(closest_points[:, 1], closest_points[:, 2], color='red', label='Closest Points on PCA Line')
        ax13.scatter(closest_points[:, 0], closest_points[:, 2], color='red', label='Closest Points on PCA Line')
    beam_vector = beam_end - beam_start
    start_point, end_point = start_end_points(closest_points, beam_mean=np.array([128, 128, 128]), dirVecBeam=beam_vector)
    if plots:
        if start_point[0] < end_point[0]:
            start_point_x = start_point[0]
            end_point_x = end_point[0]
        else:
            start_point_x = end_point[0]
            end_point_x = start_point[0]
        if start_point[1] < end_point[1]:
            start_point_y = start_point[1]
            end_point_y = end_point[1]
        else:
            start_point_y = end_point[1]
            end_point_y = start_point[1]
        if start_point[2] < end_point[2]:
            start_point_z = start_point[2]
            end_point_z = end_point[2]
        else:
            start_point_z = end_point[2]
            end_point_z = start_point[2]
        ax11.scatter(start_point[0], start_point[1], color='blue', marker='x', label='Start Point', s=100)
        ax11.scatter(end_point[0], end_point[1], color='green', marker='x', label='End Point', s=100)
        ax11.set_xlim(start_point_x-10, end_point_x+10)
        ax11.set_ylim(start_point_y-10, end_point_y+10)
        ax12.scatter(start_point[1], start_point[2], color='blue', marker='x', label='Start Point', s=100)
        ax12.scatter(end_point[1], end_point[2], color='green', marker='x', label='End Point', s=100)
        ax12.set_xlim(start_point_y-10, end_point_y+10)
        ax12.set_ylim(start_point_z-10, end_point_z+10)
        ax13.scatter(start_point[0], start_point[2], color='blue', marker='x', label='Start Point', s=100)
        ax13.scatter(end_point[0], end_point[2], color='green', marker='x', label='End Point', s=100)
        ax13.set_xlim(start_point_x-10, end_point_x+10)
        ax13.set_ylim(start_point_z-10, end_point_z+10)
    return end_point, start_point, beam_vector, dirVecTrackNorm, track_mean

# Function to get the intersection by using the closest distance between two lines.
def closest_point_on_line1(p1, d1, q1, d2):
    """
    Finds the closest point on line1 to line2 in 3D space.

    Parameters:
    p1 (numpy array): A point on line 1.
    d1 (numpy array): The direction vector of line 1.
    q1 (numpy array): A point on line 2.
    d2 (numpy array): The direction vector of line 2.

    Returns:
    numpy array: The closest point on line1 to line2.
    """
    # Convert inputs to numpy arrays for vector calculations
    p1, d1, q1, d2 = map(np.array, (p1, d1, q1, d2))

    # Compute the cross product of the direction vectors
    d1_cross_d2 = np.cross(d1, d2)
    d1_cross_d2_norm = np.linalg.norm(d1_cross_d2)  # Norm of the cross product

    # Check if lines are parallel
    if d1_cross_d2_norm == 0:
        print("The lines are parallel, so there is no unique closest point.")
        return np.array([-1,-1,-1])

    # Calculate the vector between the points on each line
    r = q1 - p1

    # Calculate the closest approach parameters for each line
    t = np.dot(np.cross(r, d2), d1_cross_d2) / d1_cross_d2_norm**2

    # Compute the closest point on line 1 using parameter t
    closest_point_line1 = p1 + t * d1

    return closest_point_line1

#######################################
# Main Function
#######################################

for energy in excitation_energies:
    for angle in cm_angles:

        filename = path+"sim_5000_"+str(energy)+"mev_"+str(angle)+"cm.root"
        f = TFile(filename)
        myTree = f.Get("SimulatedTree")
        entry = myTree.GetEntries()
        print('Reading', entry ,'entries from file', filename)

        if save_to_root:
            root_file = root.TFile("recon_sim_5000_"+str(energy)+"mev_"+str(angle)+"cm.root", "UPDATE")
            result = create_tree_and_branches("events")

        EventInfo = namedtuple('Events', ['event_id', 'verX', 'verY', 'verZ', 'dirX', 'dirY', 'dirZ', 'Eenergy', 'Elab', 'ransac', 'gmm'])
        EventInfoList = []
        exception_events = []



        for entries in myTree:
            try:
                if entries.data.event >= event_start and entries.data.event <= event_end:

                    print('Event ->', entries.data.event)
                    event_info = EventInfo(event_id=None,
                                        verX= None,
                                        verY= None,
                                        verZ= None,
                                        dirX = None,
                                        dirY = None,
                                        dirZ = None,
                                        Eenergy= None,
                                        Elab = None,
                                        ransac=None,
                                        gmm=None)

                    # Initialise dictionaries to be later written to the tree
                    ransac = {}
                    gmm = {}

                    # Get Beam Center
                    beam_center = get_beam_center(entries)

                    # Get input values from simulation
                    event_info = event_info._replace(event_id=entries.data.event,
                                                    verX= round(128.0 + entries.data.input_pos_z, 2),
                                                    verY= round(128.0 + entries.data.input_pos_x, 2),
                                                    verZ= round(128.0 + entries.data.input_pos_y, 2),
                                                    dirX = round(entries.data.input_ejectile_mom_dirZ, 2),
                                                    dirY = round(entries.data.input_ejectile_mom_dirX, 2),
                                                    dirZ = round(entries.data.input_ejectile_mom_dirY, 2),
                                                    Eenergy = round(entries.data.input_ejectile_energy, 2),
                                                    Elab = round(math.degrees(entries.data.input_theta_lab), 2)
                                                    )

                    # Get Input array and Visualize it
                    # data_array = [x, y, z, q]
                    if plots:
                        data_array, colorbars = get_data_array(beam_center, entries, event_info)
                        if debug:
                            print('Data recorded in event')
                            print(data_array)
                    else:
                        data_array = get_data_array(beam_center, entries, event_info)
                        if debug:
                            print('Data recorded in event')
                            print(data_array)

                    # Assign True Labels
                    # data_array = [x,y,z,q,true labels]
                    data_array = generate_true_labels(data_array, event_info)
                    if debug:
                        print(data_array)

                    # Get Predicted Labels from RANSAC
                    # data_array = [0-x,1-y,2-z,3-q,4-true labels, 5-ransac labels]
                    ransac_labels, fitted_models = find_multiple_lines_ransac(data_array, max_lines=10, residual_threshold=5.0, n_iterations=1000)
                    data_array = np.column_stack((data_array, ransac_labels))
                    ransac['components'] = len(np.unique(ransac_labels))

                    if plots:
                        colorbars_ransac = plot_ransac(data_array, event_info)

                    angles_ransac, low_energy_track_count_ransac = kinematics_ransac(data_array, fitted_models, True)
                    print('RANSAC Angles', angles_ransac)
                    ransac['angles'] = angles_ransac


                    #Get Prediced Labels from GMM
                    # data_array = [0-x,1-y,2-z,3-q, 4-true labels, 5-ransac labels, 6-gmm labels]
                    gmm_labels, n_comp = fit_gmm_with_bic(data_array, max_components=10)
                    data_array = np.column_stack((data_array, gmm_labels))
                    if plots:
                        colorbars_gmm = plot_gmm(data_array, event_info)
                    angles_gmm, low_energy_track_count_gmm = kinematics_gmm(data_array)
                    gmm['components'] = len(np.unique(gmm_labels))
                    gmm['angles'] = angles_gmm
                    print('GMM angles', angles_gmm)

                    # np.save('low_energy_track.npy', data_array)
                    # # Convert named tuple to a dictionary and save as JSON
                    # with open('low_energy_track.json', 'w') as f:
                    #     json.dump(event_info._asdict(), f)
                    # print(fitted_models)

                    # Function to calculate p values
                    beam_metrics, track_metrics, beam_track_metrics = calculate_cluster_metrics(data_array, beam_zone_low, beam_zone_high)
                    # print('Metrics')
                    # print(beam_metrics)
                    # print(track_metrics)
                    # print(beam_track_metrics)
                    gmm['beam_beam_metric'] = beam_metrics
                    gmm['track_track_metric'] = track_metrics
                    gmm['beam_track_metric'] = beam_track_metrics

                    # Calculate the number of components beam/track
                    unique_beam_ransac, unique_track_ransac, unique_beam_gmm, unique_track_gmm = beam_track_data(data_array)
                    ransac['beam_components'] = unique_beam_ransac
                    ransac['track_components'] = unique_track_ransac
                    gmm['beam_components'] = unique_beam_gmm
                    gmm['track_components'] = unique_track_gmm

                    ransac['ari'] = round(adjusted_rand_score(data_array[:, 4], data_array[:, 5]), 2)
                    gmm['ari'] = round(adjusted_rand_score(data_array[:, 4], data_array[:, 6]), 2)

                    # Append to the list of named tuples
                    event_info = event_info._replace(ransac=ransac)
                    event_info = event_info._replace(gmm=gmm)


                    EventInfoList.append(event_info)

                    if save_to_root:
                        print(event_info)
                        fill_event_data_to_tree(result, event_info)

                    #Final Visualization Closures
                    if plots:
                        # print('Plot here')
                        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3, wspace=0.3)
                        plt.show(block=False)
                        next_pressed = False
                        if save_python_figures:
                            next_pressed = True
                            fig.savefig('event_'+str(entries.data.event)+'.png')
                        while not next_pressed:
                            plt.waitforbuttonpress(0.1)
                        if next_pressed:
                            for colorbar in chain(colorbars, colorbars_ransac, colorbars_gmm):
                                colorbar.remove()
                            continue
                else:
                    if entries.data.event < event_start:
                        continue
                    if entries.data.event > event_end:
                        break
            except:
                print("Exception Encountered")
                exception_events.append(entries.data.event)
                continue
        print('Exited the file', exception_events)
        if save_to_root:
            print('Saving to ROOT File')
            result["tree"].Write()
            root_file.Close()
        # Define histogram parameters
        if plots:
            # print('Close Figure')
            plt.close(fig)
        all_angles_ransac = []
        all_angles_gmm = []
        all_components_ransac = []
        all_beam_components_ransac = []
        all_track_components_ransac = []
        all_beam_components_gmm = []
        all_track_components_gmm = []
        all_components_gmm = []
        all_p_value_bb = []
        all_p_value_tt = []
        all_p_value_bt = []
        all_ari_ransac = []
        all_ari_gmm = []
        for event in EventInfoList:
            # print('Elab', event.Elab)
            all_angles_ransac.extend(event.ransac['angles'].values())
            all_angles_gmm.extend(event.gmm['angles'].values())
            all_components_ransac.append(event.ransac['components'])
            all_beam_components_ransac.append(event.ransac['beam_components'])
            all_track_components_ransac.append(event.ransac['track_components'])
            all_components_gmm.append(event.gmm['components'])
            all_beam_components_gmm.append(event.gmm['beam_components'])
            all_track_components_gmm.append(event.gmm['track_components'])
            all_p_value_bb.extend(event.gmm['beam_beam_metric'].values())
            all_p_value_tt.extend(event.gmm['track_track_metric'].values())
            all_p_value_bt.extend(event.gmm['beam_track_metric'].values())
            all_ari_ransac.append(event.ransac['ari'])
            all_ari_gmm.append(event.gmm['ari'])
        if final_plots_flag:
            print('Inside Final Plots')
            fig1, axes_final = plt.subplots(3, 4, figsize=(12, 5))
            final_plots(axes_final[0,0], (0,100), 1, all_angles_ransac, 'RANSAC angles'+energy+angle)
            final_plots(axes_final[0,1], (0,100), 1, all_angles_gmm, 'GMM angles')
            final_plots(axes_final[0,2], (0,10), 1, all_components_ransac, 'K Ransac')
            final_plots(axes_final[0,3], (0,10), 1, all_components_gmm, 'K GMM')
            final_plots(axes_final[1,0], (0,10), 1, all_beam_components_ransac, 'K Ransac Beam')
            final_plots(axes_final[1,1], (0,10), 1, all_track_components_ransac, 'K Ransac Track')
            final_plots(axes_final[1,2], (0,10), 1, all_beam_components_gmm, 'K GMM Beam')
            final_plots(axes_final[1,3], (0,10), 1, all_track_components_gmm, 'K GMM Track')
            final_plots(axes_final[2,0], (0,1), 0.01, all_p_value_bb, 'Beam Beam p-value')
            final_plots(axes_final[2,1], (0,1), 0.01, all_p_value_tt, 'Track Track p-value')
            final_plots(axes_final[2,2], (0,1), 0.01, all_p_value_bt, 'Beam Track p-value')
            final_plots(axes_final[2,3], (0,1), 0.01, all_ari_ransac, 'RANSAC ARI')
            final_plots(axes_final[2,3], (0,1), 0.01, all_ari_gmm, 'GMM ARI')
            plt.show()
            plt.waitforbuttonpress()
        if save_final_data:
            print('Inside save final')
            tag = 'test'
            save_list(energy, angle, all_angles_ransac, tag+"_all_angles_ransac")
            save_list(energy, angle, all_angles_gmm, tag+"_all_angles_gmm")
            save_list(energy, angle, all_components_ransac, tag+"_all_components_ransac")
            save_list(energy, angle, all_beam_components_ransac, tag+"_all_beam_components_ransac")
            save_list(energy, angle, all_track_components_ransac, tag+"_all_track_components_ransac")
            save_list(energy, angle, all_components_gmm, tag+"_all_components_gmm")
            save_list(energy, angle, all_beam_components_gmm, tag+"_all_beam_components_gmm")
            save_list(energy, angle, all_track_components_gmm, tag+"_all_track_components_gmm")
            save_list(energy, angle, all_p_value_bb, tag+"_all_p_value_bb")
            save_list(energy, angle, all_p_value_tt, tag+"_all_p_value_tt")
            save_list(energy, angle, all_p_value_bt, tag+"_all_p_value_bt")
            save_list(energy, angle, all_ari_ransac, tag+"_all_ari_ransac")
            save_list(energy, angle, all_ari_gmm, tag+"_all_ari_gmm")
            save_list(energy, angle, exception_events, tag+"_exception_events")
            sys.exit(0)
        f.Close()