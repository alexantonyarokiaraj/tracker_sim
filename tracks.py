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
from enum import Enum
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from regularize import Regularize
from libraries import DataArray, RunParameters, VolumeBoundaries, SCAN, Optimize, FileNames, Reference, ConversionFactors
from collections import defaultdict
import warnings
from energy import Energy
import pandas as pd

np.random.seed(42)

#######################################
# Input arguments
#######################################

arguments = sys.argv
input_string = arguments[1]
split_strings = input_string.split('@')
excitation_energies=[split_strings[0]]
cm_angles=[split_strings[1]]
path = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/DATA/simulation/5000/"
event_start = int(split_strings[2])
event_end = int(split_strings[3])

plots = RunParameters.plots.value
sim = RunParameters.sim.value
debug = RunParameters.debug.value
final_plots_flag = RunParameters.final_plots_flag.value
save_final_data = RunParameters.save_final_data.value
with_missing_pads = RunParameters.with_missing_pads.value
batch_mode = RunParameters.batch_mode.value
save_to_root = RunParameters.save_to_root.value
save_python_figures = RunParameters.save_python_figures.value

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
beam_center_time = 10970.0 - beam_entrance_time  # units[ns], 7076.0
beam_center_peak_find_low = 8000  # units[ns]
beam_center_peak_find_high = 14000  # units[ns]
sig_beam_center = 70.0  # units [ns]
time_per_sample = 0.08  # units [us]
drift_velocity_volume = ConversionFactors.DRIFT_VELOCITY.value  # units [cm/us]
table = np.loadtxt(FileNames.CONVERSION_TABLE.value)
z_conversion_factor = ConversionFactors.Z_CONVERSION_FACTOR.value
x_conversion_factor = ConversionFactors.X_CONVERSION_FACTOR.value  # units[mm]
y_conversion_factor = ConversionFactors.Y_CONVERSION_FACTOR.value  # units[mm]
missing_pads_info = FileNames.MISSING_PADS.value
missed_pads = np.loadtxt(missing_pads_info)
x_pos_raw = missed_pads[:, 0]
y_pos_raw = missed_pads[:, 1]
nbins_x = ConversionFactors.NBINS_X.value
x_start_bin = ConversionFactors.X_START_BIN.value
x_end_bin = ConversionFactors.X_END_BIN.value
nbins_y = ConversionFactors.NBINS_Y.value
y_start_bin = ConversionFactors.Y_START_BIN.value
y_end_bin = ConversionFactors.Y_END_BIN.value
nbins_z = ConversionFactors.NBINS_Z.value
z_start_bin = ConversionFactors.Z_START_BIN.value
z_end_bin = ConversionFactors.Z_END_BIN.value
pixel_size_mm = 2
line_length = 100
transparency = 0.7
calibration_table = pd.read_csv(FileNames.CALIBRATION_PADS.value, sep=" ", header=None)
calibration_table.columns = ["chno", "xx", "yy", "par0", "par1", "chi"]

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
ax14 = axs[1,3]
ax15 = axs[2,3]
ax16 = axs[3,3]

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
    if ax in [ax15, ax16]:
        ax.set_xlabel('Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Charge', fontsize=12, fontweight='bold')
        ax.grid(True)

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
update_clear(ax14)
update_clear(ax15)
update_clear(ax16)

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

def get_yz_min_max(data):
    """
    Returns the minimum and maximum values for Y and Z from the data array.

    Args:
        data (np.ndarray): A NumPy array where columns represent X, Y, and Z.

    Returns:
        dict: A dictionary containing min/max values of Y and Z.
    """
    if data.shape[1] < 3:
        raise ValueError("Data must have at least three columns (X, Y, Z).")

    # Extract Y and Z columns
    y_values = data[:, DataArray.Y.value]
    z_values = data[:, DataArray.Z.value]

    # Compute min and max
    y_min, y_max = np.min(y_values), np.max(y_values)
    z_min, z_max = np.min(z_values), np.max(z_values)

    return {
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max
    }

def angle_between(v1, v2):
    """
    Calculate the angle (in degrees) between two vectors.

    Parameters:
        v1 (array-like): The first vector.
        v2 (array-like): The second vector.

    Returns:
        float: The angle between the two vectors in degrees.
    """
    # Compute cosine of the angle
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Ensure cosine is within valid range
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # Return angle in degrees
    return np.degrees(np.arccos(cos_theta))

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
            print('Center of Z->', beam_c)
        else:
            beam_c = beam_center_time - beam_entrance_time
    except:
        beam_c = beam_center_time - beam_entrance_time
    if plots:
        z_proj_nparr = np.array(z_proj_arr)
        update_clear(ax1)
        ax1.hist(z_proj_nparr, bins=50, range=(beam_center_time-1000,beam_center_time+1000), color='skyblue', alpha=0.7, label="Time")
        ax1.text(0.95, 0.95, f'Beam Center: {beam_c}',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax1.transAxes,  # Use axes coordinates
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5))
        plt.draw()
    return beam_c

def get_data_array(beam_center, entries, event_info):
    global axs
    data_points = []
    incoming_labels = []
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
                    incoming_labels.append(entries.data.CoboAsad[int(x)].trackID[int(y)])
    data = np.array(data_points)
    true_labels_sim = assign_beam_or_scattered(data, incoming_labels)
    if plots:
        # charge = data[:, DataArray.Q.value]
        charge = np.array(true_labels_sim)
        cmap = plt.cm.get_cmap("Dark2", len(np.unique(charge)))
        update_clear(ax2)
        update_clear(ax3)
        update_clear(ax4)
        colorbar1 = add_rectangles(ax2, data[:, DataArray.X.value:DataArray.Z.value + 1], charge, cmap, proj = 'xy', colorbarFlag=False, discrete=True)
        colorbar2 = add_rectangles(ax3, data[:, DataArray.X.value:DataArray.Z.value + 1], charge, cmap, proj = 'yz', colorbarFlag=False, discrete=True)
        colorbar3 = add_rectangles(ax4, data[:, DataArray.X.value:DataArray.Z.value + 1], charge, cmap, proj = 'xz', colorbarFlag=False, discrete=True)
        num_data_points = len(data[:, DataArray.X.value])
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
        return data, incoming_labels, [colorbar1, colorbar2, colorbar3]
    # print('Checking plot return')
    return data, incoming_labels

def generate_true_labels(data_array, event_info):
    y_values = data_array[:, DataArray.Y.value]
    labels = np.where((y_values >= VolumeBoundaries.BEAM_ZONE_MIN.value) & (y_values < VolumeBoundaries.BEAM_ZONE_MAX.value), 0, 1)
    data_array = np.column_stack((data_array, labels))
    if plots:
        cmap = plt.cm.get_cmap("Dark2", len(np.unique(labels)))
        update_clear(ax5)
        update_clear(ax6)
        update_clear(ax7)
        colorbar4 = add_rectangles(ax5, data_array[:, DataArray.X.value:DataArray.Z.value + 1], labels, cmap, proj = 'xy', colorbarFlag=False, discrete=True)
        colorbar5 = add_rectangles(ax6, data_array[:, DataArray.X.value:DataArray.Z.value + 1], labels, cmap, proj = 'yz', colorbarFlag=False, discrete=True)
        colorbar6 = add_rectangles(ax7, data_array[:, DataArray.X.value:DataArray.Z.value + 1], labels, cmap, proj = 'xz', colorbarFlag=False, discrete=True)
        num_data_points = len(data_array[:, DataArray.X.value])
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

def plot_ransac(data_array, event_info):
    labels = data_array[:, DataArray.ransac_labels.value]
    cmap = plt.cm.get_cmap("Dark2", len(np.unique(labels)))
    update_clear(ax8)
    update_clear(ax9)
    update_clear(ax10)
    colorbar4 = add_rectangles(ax8, data_array[:, DataArray.X.value:DataArray.Z.value + 1], labels, cmap, proj = 'xy', colorbarFlag=True, discrete=True)
    colorbar5 = add_rectangles(ax9, data_array[:, DataArray.X.value:DataArray.Z.value + 1], labels, cmap, proj = 'yz', colorbarFlag=True, discrete=True)
    colorbar6 = add_rectangles(ax10, data_array[:, DataArray.X.value:DataArray.Z.value + 1], labels, cmap, proj = 'xz', colorbarFlag=True, discrete=True)
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

def return_threshold_lines(start_point, direction_vector):
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

def extend_line_based_on_reference(start_point, end_point, start_point_full, end_point_full, inter, extra_start: float = float(Reference.RANGE_EXTEND.value)):
    """
    Extends a 3D line segment:
    - The start point moves back by `extra_start` mm.
    - The end point moves forward so that the new total length is at least `20 mm` longer than the reference line.

    Parameters:
    - start_point (numpy array): Start point of the primary line (x, y, z).
    - end_point (numpy array): End point of the primary line (x, y, z).
    - start_point_full (numpy array): Start point of the reference line (x, y, z).
    - end_point_full (numpy array): End point of the reference line (x, y, z).
    - extra_start (float): Fixed extension length for the start point (default is 20 mm).

    Returns:
    - numpy array: A (2,3) array with the new extended start and end points.
    """

    # Convert to numpy arrays
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    start_point_full = np.array(start_point_full)
    end_point_full = np.array(end_point_full)

    # Compute the direction vector and normalize
    direction = end_point - start_point
    original_length = np.linalg.norm(direction)

    if original_length == 0:
        raise ValueError("Start and end points are the same. Cannot define a line.")

    unit_direction = direction / original_length

    # Compute the reference line length
    reference_length = np.linalg.norm(end_point_full - start_point_full)

    # Compute new total length (at least 20mm longer than reference)
    new_total_length = reference_length + extra_start
    extra_end = new_total_length - original_length  # Remaining extension for the end

    if extra_end < 0:
        raise ValueError("Reference line is too short to extend properly.")

    # Extend start and end points
    extended_start = start_point - extra_start * unit_direction
    extended_end = end_point + extra_end * unit_direction

    # Return as a (2,3) array
    return np.vstack((inter, extended_end))

def plot_energy_distributions(ax, r2d, sd, ran_end_, en_end_, ran_max_, en_max_,
                             charge_profile_x, charge_profile_y,
                             charge_profile_x_s, charge_profile_y_s):
    """
    Plots the smoothed and unsmoothed energy distributions as a function of position on the given Axes object.
    Adds horizontal lines at the position of the maximum energy and the last intersection point.

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object on which to plot.
        r2d (float): 2D distance of the final bin or peak from the start of the line.
        sd (float): Vertical (Z-axis) distance of the final bin or peak from the start of the line.
        ran_end_ (float): Position of the last intersection point between the energy curve and threshold.
        en_end_ (float): Energy value at the last intersection point.
        ran_max_ (float): Position of the maximum energy value in the interpolated distribution.
        en_max_ (float): Maximum energy value in the interpolated distribution.
        charge_profile_x (numpy array): Positions of the unsmoothed energy bins.
        charge_profile_y (numpy array): Unsmoothed energy values.
        charge_profile_x_s (numpy array): Positions of the smoothed energy bins.
        charge_profile_y_s (numpy array): Smoothed energy values.
    """
    update_clear(ax)
    # Plot the unsmoothed energy distribution
    ax.plot(charge_profile_x, charge_profile_y, 'b-', label="Unsmoothed Energy", alpha=0.6)

    # Plot the smoothed energy distribution
    ax.plot(charge_profile_x_s, charge_profile_y_s, 'r-', label="Smoothed Energy", linewidth=2)

    # Add horizontal lines for key positions
    ax.axvline(x=ran_max_, color='g', linestyle='--', label=f"Max Energy Position: {ran_max_:.2f}")
    ax.axvline(x=ran_end_, color='m', linestyle='--', label=f"Last Intersection Position: {ran_end_:.2f}")

def filter_track_data(cut_data_charge, track_above):
    """
    Filters the cut_data_charge array based on track position.

    Parameters:
    cut_data_charge (numpy.ndarray): 2D NumPy array containing the data.

    Returns:
    numpy.ndarray: Filtered array based on track position.
    """
    if track_above:
        filtered_data = cut_data_charge[cut_data_charge[:, DataArray.Y.value] > VolumeBoundaries.BEAM_ZONE_MAX.value]
    else:
        filtered_data = cut_data_charge[cut_data_charge[:, DataArray.Y.value] < VolumeBoundaries.BEAM_ZONE_MIN.value]

    return filtered_data

def kinematics_ransac(data, fitted_models, useLineModelND):
    # Define the beam line endpoints
    data = data[data[:, DataArray.ransac_labels.value] != 20]
    data = add_filters(data, model= int(DataArray.ransac_labels.value))
    lab_angles_initial = {}
    intersections_initial = {}
    phi_angles_initial = {}
    start_point_initial = {}
    end_point_initial = {}
    ranges_initial = {}
    ranges_final = {}
    energy_initial = {}


    ransac_labels = np.unique(data[:, DataArray.ransac_labels.value])
    for label in ransac_labels:
        # Get the points corresponding to the current label
        cluster_data = data[data[:, DataArray.ransac_labels.value] == label]

        # Check if the cluster size is greater than 10
        if cluster_data.shape[0] <= 10:
            continue  # Skip this cluster if it has 10 or fewer points

        # Calculate the mean y of the cluster
        mean_y = np.mean(cluster_data[:, DataArray.Y.value])
        # Process clusters either above or below the beam line
        if (mean_y >= VolumeBoundaries.BEAM_ZONE_MAX.value or mean_y < VolumeBoundaries.BEAM_ZONE_MIN.value) and cluster_data.shape[0] >= 10:
            mask = (
                (cluster_data[:, DataArray.scattered_track.value] == 1) &
                (cluster_data[:, DataArray.track_inside_volume.value] == 1) &
                (cluster_data[:, DataArray.vertex_inside_volume.value] == 1) &
                ((cluster_data[:, DataArray.side_of_track.value] == 1) | (cluster_data[:, DataArray.side_of_track.value] == -1)) &  # Column : (side_flag_col) = 1 or -1
                ((cluster_data[:, DataArray.closest_track.value] == 1) | (cluster_data[:, DataArray.closest_track.value] == -1)) & # Column: (proximity_flag_col) = 1 or -1
                (cluster_data[:, DataArray.end_point_above_beam_zone.value] == 1) # Column 12: Track End point above the beam zone
            )
            cut_data = cluster_data[mask, :3]
            cut_data_charge = cluster_data[mask, :4]
            if cut_data.size > 0:
                # Fill the lab angles and intersections based on first PCA fit.
                end_point_full, start_point_full, beam_vector_full, dirVecTrackNorm_full, track_mean_full, closest_points_full = get_directions(cut_data)
                track_vector_full = end_point_full - start_point_full
                intersection_point_full = closest_point_on_line1(start_point_full, track_vector_full, np.array([0,128,128]), beam_vector_full)
                distances_from_start = np.linalg.norm(closest_points_full - start_point_full, axis=1)
                mask_beta = (distances_from_start >= 0) & (distances_from_start <= Optimize.BETA.value)
                filtered_data_beta = cut_data[mask_beta, :]
                if len(filtered_data_beta) > 1:
                    end_point_beta, start_point_beta, beam_vector_beta, dirVecTrackNorm_beta, track_mean_beta, closest_points_beta = get_directions(filtered_data_beta)
                    track_vector_beta = end_point_beta - start_point_beta
                    lab_angle_beta = angle_between(track_vector_beta, beam_vector_beta)
                    phi_angle_beta = calculate_phi_angle(track_vector_beta, beam_vector_beta)
                    intersection_point_beta = closest_point_on_line1(start_point_beta, track_vector_beta, np.array([0,128,128]), beam_vector_beta)
                    lab_angles_initial[label] = round(lab_angle_beta, 2)
                    intersections_initial[label] = intersection_point_beta
                    start_point_initial[label] = start_point_beta
                    end_point_initial[label] = end_point_beta
                    phi_angles_initial[label] = phi_angle_beta
                    endpts = extend_line_based_on_reference(start_point_beta, end_point_beta, start_point_full, end_point_full, intersection_point_beta, extra_start=float(Reference.RANGE_EXTEND.value))
                    en = Energy(cut_data_charge, endpts, calibration_table)
                    new_position, fit_energy_, line_vector_start_3d, unit_vector_3d, line_length_2d, line_vector_end_3d, histogram_array_new = en.calculate_profiles()
                    if RunParameters.optimize_alpha.value:
                        ranges = {}
                        alpha_values = np.linspace(Optimize.ALPHA_RANGE_LOW.value, Optimize.ALPHA_RANGE_HIGH.value, Optimize.ALPHA_STEPS.value)

                        for alpha in alpha_values:
                            _, _, ran_end_, _, ran_max_, _, _, _, _, _ = en.energy_weighted(alpha, new_position, fit_energy_, line_vector_start_3d, unit_vector_3d, line_length_2d, line_vector_end_3d, histogram_array_new)
                            ranges[alpha] = (ran_end_, ran_max_)
                        ranges_initial[label] = ranges
                    r2d, sd, ran_end_, en_end_, ran_max_, en_max_, charge_profile_x, charge_profile_y, charge_profile_x_s, charge_profile_y_s = en.energy_weighted(Optimize.ALPHA.value, new_position, fit_energy_, line_vector_start_3d, unit_vector_3d, line_length_2d, line_vector_end_3d, histogram_array_new)
                    ranges_final[label] = ran_end_
                    if plots:
                        plot_lines(track_mean_full, dirVecTrackNorm_full, start_point_full, end_point_full, intersection_point_full, closest_points_full, ax8, ax9, ax10, color='blue', s=400)
                        plot_lines(track_mean_beta, dirVecTrackNorm_beta, start_point_beta, end_point_beta, intersection_point_beta, closest_points_beta, ax8, ax9, ax10, color='green', s=200)
                        plot_lines(track_mean_beta, dirVecTrackNorm_beta, endpts[0, :], endpts[1, :], intersection_point_beta, closest_points_beta, ax8, ax9, ax10, color='red', s=100)
                        plot_energy_distributions(ax15, r2d, sd, ran_end_, en_end_, ran_max_, en_max_,
                             charge_profile_x, charge_profile_y,
                             charge_profile_x_s, charge_profile_y_s)
                        if not useLineModelND:
                            threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2 = return_threshold_lines(track_mean_full, dirVecTrackNorm_full)
                            plot_threshold_lines(track_mean_full, dirVecTrackNorm_full, threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2)
                        else:
                            model_params = fitted_models[label].params
                            threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2 = return_threshold_lines(np.array(model_params[0]), np.array(model_params[1]))
                            plot_threshold_lines(np.array(model_params[0]), np.array(model_params[1]), threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2)
            else:
                end_point, start_point, beam_vector, dirVecTrackNorm, track_mean, closest_points = get_directions(cluster_data[:, :3])
                track_vector = end_point - start_point
                intersection_point = closest_point_on_line1(start_point, track_vector, np.array([0,128,128]), beam_vector)
                if plots:
                    plot_lines(track_mean, dirVecTrackNorm, start_point, end_point, intersection_point, closest_points, ax8, ax9, ax10, color='black', s = 500)
                    if not useLineModelND:
                        threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2 = return_threshold_lines(track_mean, dirVecTrackNorm)
                        plot_threshold_lines(track_mean, dirVecTrackNorm, threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2)
                    else:
                        model_params = fitted_models[label].params
                        threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2 = return_threshold_lines(np.array(model_params[0]), np.array(model_params[1]))
                        plot_threshold_lines(np.array(model_params[0]), np.array(model_params[1]), threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2)
    return lab_angles_initial, intersections_initial, start_point_initial, end_point_initial, phi_angles_initial, ranges_initial, ranges_final

# Function to plot RANSAC threshold lines
def plot_threshold_lines(start_point, direction_vector, threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2):

    line_start = start_point - 50 * direction_vector
    line_end = start_point + 50 * direction_vector

    # Threshold line projections for the first perpendicular vector
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
            linestyle="--")
    ax8.plot(xy_threshold_above1[:, 0], xy_threshold_above1[:, 1], 'r--', label='Threshold +5mm')
    ax8.plot(xy_threshold_below1[:, 0], xy_threshold_below1[:, 1], 'g--', label='Threshold -5mm')
    ax8.plot(xy_threshold_above2[:, 0], xy_threshold_above2[:, 1], 'r--')
    ax8.plot(xy_threshold_below2[:, 0], xy_threshold_below2[:, 1], 'g--')
    ax9.plot([line_start[1], line_end[1]],
            [line_start[2], line_end[2]],
            linestyle="--")
    ax9.plot(yz_threshold_above1[:, 0], yz_threshold_above1[:, 1], 'r--', label='Threshold +5mm')
    ax9.plot(yz_threshold_below1[:, 0], yz_threshold_below1[:, 1], 'g--', label='Threshold -5mm')
    ax9.plot(yz_threshold_above2[:, 0], yz_threshold_above2[:, 1], 'r--')
    ax9.plot(yz_threshold_below2[:, 0], yz_threshold_below2[:, 1], 'g--')
    ax10.plot([line_start[0], line_end[0]],
            [line_start[2], line_end[2]],
            linestyle="--")
    ax10.plot(xz_threshold_above1[:, 0], xz_threshold_above1[:, 1], 'r--', label='Threshold +5mm')
    ax10.plot(xz_threshold_below1[:, 0], xz_threshold_below1[:, 1], 'g--', label='Threshold -5mm')
    ax10.plot(xz_threshold_above2[:, 0], xz_threshold_above2[:, 1], 'r--')
    ax10.plot(xz_threshold_below2[:, 0], xz_threshold_below2[:, 1], 'g--')
    plt.draw

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
        n_samples = features.shape[0]
        if n_components > features.shape[0]:
            break  # Exit loop early if n_components exceeds n_samples
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
    responsibilities = best_gmm.predict_proba(features)

    return best_labels, best_n_components, responsibilities

# Function to plot the GMM assigned clusters
def plot_gmm(data_array, event_info, color = 'blue', size=50):
    labels = data_array[:, DataArray.merge_p_val.value]
    cmap = plt.cm.get_cmap("Dark2", len(np.unique(labels)))
    update_clear(ax11)
    update_clear(ax12)
    update_clear(ax13)
    colorbar7 = add_rectangles(ax11, data_array[:, DataArray.X.value:DataArray.Z.value + 1], labels, cmap, proj = 'xy', colorbarFlag=True, discrete=True)
    colorbar8 = add_rectangles(ax12, data_array[:, DataArray.X.value:DataArray.Z.value + 1], labels, cmap, proj = 'yz', colorbarFlag=True, discrete=True)
    colorbar9 = add_rectangles(ax13, data_array[:, DataArray.X.value:DataArray.Z.value + 1], labels, cmap, proj = 'xz', colorbarFlag=True, discrete=True)
     # Place the number of data points in the top right corner
    print('Printing input vertices', event_info.verX, event_info.verY, event_info.verZ, event_info.dirX, event_info.dirY, event_info.dirZ, line_length)
    ax11.plot([event_info.verX, event_info.verX + line_length * event_info.dirX],[event_info.verY, event_info.verY + line_length * event_info.dirY],
                color=color, alpha=transparency)
    ax11.scatter(event_info.verX, event_info.verY, color=color, edgecolor='black', s=size, zorder=3)  # Circle for vertex
    ax12.plot([event_info.verY, event_info.verY + line_length * event_info.dirY],[event_info.verZ, event_info.verZ + line_length * event_info.dirZ],
                 color=color, alpha=transparency)
    ax12.scatter(event_info.verY, event_info.verZ, color=color, edgecolor='black', s=size, zorder=3)  # Circle for vertex
    ax13.plot([event_info.verX, event_info.verX + line_length * event_info.dirX],[event_info.verZ, event_info.verZ + line_length * event_info.dirZ],
                 color=color, alpha=transparency)
    ax13.scatter(event_info.verX, event_info.verZ, color=color, edgecolor='black', s=size, zorder=3)  # Circle for vertex
    plt.draw()
    # print('GMM colorbar', [colorbar7, colorbar8, colorbar9])
    return [colorbar7, colorbar8, colorbar9]

# Function to plot the kinematics of GMM Clusters
def kinematics_gmm(data, responsibilities, event_info):
    """
    Analyze clusters based on GMM labels to find direction vectors, intersections with a beam line,
    and angles. Plot the clusters with their fitted line in XY projection.

    Parameters:
    - data (np.ndarray): Input data array with columns [x, y, z, q, true labels, ransac labels, gmm labels, dbscan labels, merge labels].

    Returns:
    - intersections (dict): Dictionary of intersections for qualifying clusters.
    - angles (dict): Dictionary of angles (in degrees) between the cluster direction vectors and the beamline.
    """

    intersections_initial = {}
    lab_angles_initial = {}
    phi_angles_initial = {}
    lab_angles_minimize = {}
    start_point_initial = {}
    end_point_initial = {}
    closest_threshold_dict = {}
    closest_angle_dict = {}
    ranges_initial = {}
    ranges_final = {}
    intersections_final = {}

    gmm_labels = np.unique(data[:, DataArray.merge_cdist.value])
    for label in gmm_labels:
        cluster_data = data[data[:, DataArray.merge_cdist.value] == label]
        if len(cluster_data) < 10:
            continue  # Skip clusters with fewer than 10 points
        # Calculate the mean of Y coordinate
        mean_y = np.mean(cluster_data[:, DataArray.Y.value])
        if (mean_y >= VolumeBoundaries.BEAM_ZONE_MAX.value or mean_y < VolumeBoundaries.BEAM_ZONE_MIN.value) and cluster_data.shape[0] >= 10:
            mask = (
                (cluster_data[:, DataArray.scattered_track.value] == 1) &
                (cluster_data[:, DataArray.track_inside_volume.value] == 1) &
                ((cluster_data[:, DataArray.side_of_track.value] == 1) | (cluster_data[:, DataArray.side_of_track.value] == -1)) &
                (cluster_data[:, DataArray.end_point_above_beam_zone.value] == 1)
            )
            cut_data = cluster_data[mask, :3]
            # Determine if the track is above or below
            cut_data_track = cluster_data[mask, :]
            track_above = np.all(cut_data_track[:, DataArray.side_of_track.value] == 1)
            track_below = np.all(cut_data_track[:, DataArray.side_of_track.value] == -1)
            if track_above:
                side_flag = True
            if track_below:
                side_flag = False
            if cut_data.size > 0:
                # Fill the lab angles and intersections based on first PCA fit.
                end_point_full, start_point_full, beam_vector_full, dirVecTrackNorm_full, track_mean_full, closest_points_full = get_directions(cut_data)
                track_vector_full = end_point_full - start_point_full
                intersection_point_full = closest_point_on_line1(start_point_full, track_vector_full, np.array([0,128,128]), beam_vector_full)
                distances_from_start = np.linalg.norm(closest_points_full - start_point_full, axis=1)
                mask_beta = (distances_from_start >= 0) & (distances_from_start <= Optimize.BETA.value)
                filtered_data_beta = cut_data[mask_beta, :]
                if len(filtered_data_beta) > 1:
                    end_point_beta, start_point_beta, beam_vector_beta, dirVecTrackNorm_beta, track_mean_beta, closest_points_beta = get_directions(filtered_data_beta)
                    track_vector_beta = end_point_beta - start_point_beta
                    lab_angle_beta = angle_between(track_vector_beta, beam_vector_beta)
                    phi_angle_beta = calculate_phi_angle(track_vector_beta, beam_vector_beta)
                    intersection_point_beta = closest_point_on_line1(start_point_beta, track_vector_beta, np.array([0,128,128]), beam_vector_beta)
                    # lab_angles_initial[label] = round(lab_angle_beta, 2)
                    # intersections_initial[label] = intersection_point_beta
                    # start_point_initial[label] = start_point_beta
                    # end_point_initial[label] = end_point_beta
                    # phi_angles_initial[label] = phi_angle_beta
                if plots:
                    plot_lines(track_mean_full, dirVecTrackNorm_full, start_point_full, end_point_full, intersection_point_full, closest_points_full, ax11, ax12, ax13, color='blue', s=400)
                    plot_lines(track_mean_beta, dirVecTrackNorm_beta, start_point_beta, end_point_beta, intersection_point_beta, closest_points_beta, ax11, ax12, ax13, color='green', s=200)
                # print('Track inside Volume', start_point, end_point, intersection_point, np.unique(cluster_data[:, 10]), np.unique(cluster_data[:, 11]), np.unique(cluster_data[:, 12]))
            else:
                end_point, start_point, beam_vector, dirVecTrackNorm, track_mean, closest_points = get_directions(cluster_data[:, :3])
                track_vector = end_point - start_point
                intersection_point = closest_point_on_line1(start_point, track_vector, np.array([0,128,128]), beam_vector)
                # print('Track not inside Volume', start_point, end_point, intersection_point, np.unique(cluster_data[:, 10]), np.unique(cluster_data[:, 11]), np.unique(cluster_data[:, 12]))
                # if 0 in np.unique(cluster_data[:, 12]):
                #     print('Track End point inside')
                if plots:
                    plot_lines(track_mean, dirVecTrackNorm, start_point, end_point, intersection_point, closest_points, ax11, ax12, ax13)

            # Start Minimization
            lab_angles_resp = {}
            range_1 = np.linspace(0, 0.01, 50)
            range_2 = np.linspace(0.01, 0.1, 50)
            range_3 = np.linspace(0.1, 0.2, 50)
            range_4 = np.linspace(0.2, 0.3, 50)
            range_5 = np.linspace(0.3, 0.4, 50)
            range_6 = np.linspace(0.4, 0.5, 50)
            range_7 = np.linspace(0.5, 1, 10)
            # Concatenate all ranges
            beam_zone_mask = (data[:, DataArray.Y.value] >= VolumeBoundaries.BEAM_ZONE_MIN.value-2) & (data[:, DataArray.Y.value] <= VolumeBoundaries.BEAM_ZONE_MAX.value+2)
            labels_for_current_label = data[:, DataArray.merge_cdist.value] == label
            not_belonging_to_label = ~labels_for_current_label
            inside_beam_zone_not_label = beam_zone_mask & not_belonging_to_label
            res = np.concatenate([range_1, range_2, range_3, range_4, range_5, range_6, range_7])
            gmm_indices = np.where(data[:, DataArray.merge_cdist.value] == label)[0]
            gmm_labels_raw = np.unique(data[gmm_indices, DataArray.gmm_labels.value])
            gmm_labels_raw = np.array(gmm_labels_raw, dtype=int)
            if not RunParameters.optimize_gamma.value:
                res = [Optimize.GAMMA.value]
            cut_data = cluster_data[mask, :3]
            cut_data_charge = cluster_data[mask, :4]
            if cut_data.size > 0:
                end_point_fnew, start_point_fnew, beam_vector_fnew, dirVecTrackNorm_fnew, track_mean_fnew, closest_points_fnew = get_directions(cut_data)
                distances_from_start_fnew = np.linalg.norm(closest_points_fnew - start_point_fnew, axis=1)
                mask_beta_fnew = (distances_from_start_fnew >= 0) & (distances_from_start_fnew <= Optimize.BETA.value)
                for res_threshold in res:
                    responsibility_threshold = res_threshold
                    responsibility_mask = np.any(responsibilities[:, gmm_labels_raw] > responsibility_threshold, axis=1)
                    final_mask = inside_beam_zone_not_label & responsibility_mask
                    data_for_angle = np.vstack((cut_data[mask_beta_fnew, :3], data[final_mask, :3]))
                    end_point_resp, start_point_resp, beam_vector_resp, dirVecTrackNorm_resp, track_mean_resp, closest_points_resp = get_directions(data_for_angle)
                    track_vector_resp = end_point_resp - start_point_resp
                    lab_angle_p = angle_between(track_vector_resp, beam_vector_resp)
                    lab_angles_resp[res_threshold] = round(lab_angle_p, 2)
                lab_angles_minimize[label] = lab_angles_resp
                closest_threshold, closest_angle = min(lab_angles_resp.items(), key=lambda item: abs(item[1] - event_info.Elab))
                closest_threshold_dict[label] = closest_threshold
                closest_angle_dict[label] = closest_angle
                # responsibility_threshold = closest_threshold
                responsibility_threshold = Optimize.GAMMA.value
                responsibility_mask = np.any(responsibilities[:, gmm_labels_raw] > responsibility_threshold, axis=1)
                final_mask = inside_beam_zone_not_label & responsibility_mask
                data_for_angle = np.vstack((cut_data[mask_beta_fnew, :3], data[final_mask, :3]))
                end_point_full, start_point_full, beam_vector_full, dirVecTrackNorm_full, track_mean_full, closest_points_full = get_directions(cut_data)
                track_vector_full = end_point_full - start_point_full
                intersection_point_full = closest_point_on_line1(start_point_full, track_vector_full, np.array([0,128,128]), beam_vector_full)
                end_point_resp, start_point_resp, beam_vector_resp, dirVecTrackNorm_resp, track_mean_resp, closest_points_resp = get_directions(data_for_angle)
                track_vector_resp = end_point_resp - start_point_resp
                intersection_point_resp = closest_point_on_line1(start_point_resp, track_vector_resp, np.array([0,128,128]), beam_vector_resp)
                # intersections_final[label] = intersection_point_resp
                lab_angle_resp = angle_between(track_vector_resp, beam_vector_resp)
                phi_angle_resp = calculate_phi_angle(track_vector_resp, beam_vector_resp)
                lab_angles_initial[label] = round(lab_angle_resp, 2)
                intersections_initial[label] = intersection_point_resp
                start_point_initial[label] = start_point_resp
                end_point_initial[label] = end_point_resp
                phi_angles_initial[label] = phi_angle_resp

                endpts = extend_line_based_on_reference(start_point_resp, end_point_resp, start_point_full, end_point_full, intersection_point_resp, extra_start=float(Reference.RANGE_EXTEND.value))
                en = Energy(filter_track_data(cut_data_charge, side_flag), endpts, calibration_table)
                new_position, fit_energy_, line_vector_start_3d, unit_vector_3d, line_length_2d, line_vector_end_3d, histogram_array_new = en.calculate_profiles()
                if RunParameters.optimize_alpha.value:
                    ranges = {}
                    alpha_values = np.linspace(Optimize.ALPHA_RANGE_LOW.value, Optimize.ALPHA_RANGE_HIGH.value, Optimize.ALPHA_STEPS.value)

                    for alpha in alpha_values:
                        _, _, ran_end_, _, ran_max_, _, _, _, _, _ = en.energy_weighted(alpha, new_position, fit_energy_, line_vector_start_3d, unit_vector_3d, line_length_2d, line_vector_end_3d, histogram_array_new)
                        ranges[alpha] = (ran_end_, ran_max_)
                    ranges_initial[label] = ranges
                r2d, sd, ran_end_, en_end_, ran_max_, en_max_, charge_profile_x, charge_profile_y, charge_profile_x_s, charge_profile_y_s = en.energy_weighted(Optimize.ALPHA.value, new_position, fit_energy_, line_vector_start_3d, unit_vector_3d, line_length_2d, line_vector_end_3d, histogram_array_new)
                print('Energy Loss Profile')
                print(((en_max_-en_end_)-(ran_end_-ran_max_))/en_max_)
                ranges_final[label] = ran_end_
                print('Lowest Angle, Threshold', round(angle_between(track_vector_resp, beam_vector_resp), 2), closest_threshold*100)
                if plots:
                    plot_lines(track_mean_resp, dirVecTrackNorm_resp, endpts[0, :], endpts[1, :], intersection_point_resp, closest_points_resp, ax11, ax12, ax13, color='red', s=100)
                    plot_energy_distributions(ax16, r2d, sd, ran_end_, en_end_, ran_max_, en_max_,
                            charge_profile_x, charge_profile_y,
                            charge_profile_x_s, charge_profile_y_s)
                    ax11.scatter(data[final_mask, 0]+1, data[final_mask, 1]+1, marker = 'o')
                    ax12.scatter(data[final_mask, 1]+1, data[final_mask, 2]+1, marker = 'o')
                    ax13.scatter(data[final_mask, 0]+1, data[final_mask, 2]+1, marker = 'o')
    return lab_angles_initial, intersections_initial, lab_angles_minimize, start_point_initial, end_point_initial, closest_threshold_dict, closest_angle_dict, phi_angles_initial, data, ranges_initial, ranges_final

# Function to plot the kinematics of GMM Clusters
def calculate_beta(data, model = None):
    if model == DataArray.ransac_labels.value:
        data = add_filters(data, model= int(DataArray.ransac_labels.value))
    lab_angles_beta = {}
    track_labels = np.unique(data[:, model])
    for label in track_labels:
        cluster_data = data[data[:, model] == label]
        if len(cluster_data) < 10:
            continue  # Skip clusters with fewer than 10 points
        # Calculate the mean of Y coordinate
        mean_y = np.mean(cluster_data[:, DataArray.Y.value])
        if (mean_y >= VolumeBoundaries.BEAM_ZONE_MAX.value or mean_y < VolumeBoundaries.BEAM_ZONE_MIN.value) and cluster_data.shape[0] >= 10:
            # Mask for selecting tracks
            mask = (
                    (cluster_data[:, DataArray.scattered_track.value] == 1) &
                    (cluster_data[:, DataArray.track_inside_volume.value] == 1) &
                    ((cluster_data[:, DataArray.side_of_track.value] == 1) | (cluster_data[:, DataArray.side_of_track.value] == -1)) &
                    (cluster_data[:, DataArray.end_point_above_beam_zone.value] == 1)
                )
            cut_data = cluster_data[mask, :3]
            if cut_data.size > 0:
                end_point, start_point, beam_vector, dirVecTrackNorm, track_mean, closest_points = get_directions(cut_data)
                distances_from_start = np.linalg.norm(closest_points - start_point, axis=1)
                angle_beta_dict = {}
                for beta in range(Optimize.BETA_RANGE_LOW.value, Optimize.BETA_RANGE_HIGH.value, Optimize.BETA_STEPS.value):
                    mask_beta = (distances_from_start >= 0) & (distances_from_start <= beta)
                    filtered_data = cut_data[mask_beta, :]
                    if len(filtered_data) > 1:
                        end_point_beta, start_point_beta, beam_vector_beta, dirVecTrackNorm_beta, track_mean_beta, closest_points_beta = get_directions(filtered_data)
                        track_vector_beta = end_point_beta - start_point_beta
                        lab_angle_beta = angle_between(track_vector_beta, beam_vector_beta)
                        angle_beta_dict[beta] = lab_angle_beta
                        if plots:
                            if model == DataArray.ransac_labels.value:
                                plot_beta_points(start_point_beta, end_point_beta, ax8, ax9, ax10)
                            if model == DataArray.merge_cdist.value:
                                plot_beta_points(start_point_beta, end_point_beta, ax11, ax12, ax13)
                lab_angles_beta[label] = angle_beta_dict
    return lab_angles_beta

# Function to add plot lines
def plot_beta_points(start_point, end_point, ax11, ax12, ax13):
    ax11.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], label=f'Fitted Line')
    ax12.plot([start_point[1], end_point[1]], [start_point[2], end_point[2]], label=f'Fitted Line')
    ax13.plot([start_point[0], end_point[0]], [start_point[2], end_point[2]], label=f'Fitted Line')
    ax11.scatter(start_point[0], start_point[1], color='blue', marker='x', label='Start Point', s=200)
    ax11.scatter(end_point[0], end_point[1], color='green', marker='x', label='End Point', s=100)
    ax12.scatter(start_point[1], start_point[2], color='blue', marker='x', label='Start Point', s=200)
    ax12.scatter(end_point[1], end_point[2], color='green', marker='x', label='End Point', s=100)


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
    ransac_labels = data_array[:, DataArray.ransac_labels.value]
    gmm_labels = data_array[:, DataArray.gmm_labels.value]
    y_values = data_array[:, DataArray.Y.value]  # Assuming the second column is y

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

            if VolumeBoundaries.BEAM_ZONE_MIN.value <= mean_y < VolumeBoundaries.BEAM_ZONE_MAX.value:
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

    # non_constant_features = data

    if np.any(np.std(data, axis=0) == 0):
        print("Warning: Constant features detected.")
        # non_constant_features = data[:, np.std(data, axis=0) > 0]

    if len(np.unique(data, axis=0)) < len(data):
        print("Warning: Duplicate rows detected.")

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print("Warning: Data contains NaN or inf values.")


    # Catch the specific warning related to constant features in PCA
    with warnings.catch_warnings():

        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Apply PCA with 1 component
        pca = PCA(n_components=1)
        pca.fit(data)

    dirVecTrack = pca.components_[0]
    dirVecTrackNorm = dirVecTrack / np.linalg.norm(dirVecTrack)
    track_mean = pca.mean_
    closest_points = find_closest_points_on_line(data, dirVecTrack, track_mean)
    beam_vector = beam_end - beam_start
    start_point, end_point = start_end_points(closest_points, beam_mean=np.array([128, 128, 128]), dirVecBeam=beam_vector)
    # Calculate distances of each point's y-coordinate from the beam zone
    dist_start = min(abs(start_point[1] - VolumeBoundaries.BEAM_CENTER.value),
                    abs(start_point[1] - VolumeBoundaries.BEAM_CENTER.value))
    dist_end = min(abs(end_point[1] - VolumeBoundaries.BEAM_CENTER.value),
                abs(end_point[1] - VolumeBoundaries.BEAM_CENTER.value))
    # Swap points if end_point is closer to the beam zone than start_point
    if dist_end < dist_start:
        start_point, end_point = end_point, start_point
    return end_point, start_point, beam_vector, dirVecTrackNorm, track_mean, closest_points

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

# Function to add filters to the data array
def add_filters(data, model):
    # - data (np.ndarray): Input data array with columns [x, y, z, q, true labels, ransac labels, gmm labels, dbscan labels, merge labels].

    # Initialize additional columns
    beam_track_flag_col = np.zeros((data.shape[0], 1))    # Column 8
    volume_check_flag_col = np.zeros((data.shape[0], 1))  # Column 9
    intersection_flag_col = np.zeros((data.shape[0], 1))  # Column 10
    side_flag_col = np.zeros((data.shape[0], 1))          # Column 11
    proximity_flag_col = np.zeros((data.shape[0], 1))     # Column 12
    end_point_flag_col = np.zeros((data.shape[0], 1))     # Column 13

    # Unique GMM labels identify tracks
    gmm_labels = np.unique(data[:, model])  # Assuming GMM labels are in the 7th column

    side_tracks = {'above': [], 'below': []}

    for label in gmm_labels:
        # Select track data and indices
        cluster_data = data[data[:, model] == label]
        track_indices = np.where(data[:, model] == label)[0]

        # Calculate mean_y for determining beam or scattered track (Column 8)
        mean_y = np.mean(cluster_data[:, DataArray.Y.value])
        if VolumeBoundaries.BEAM_ZONE_MIN.value <= mean_y <= VolumeBoundaries.BEAM_ZONE_MAX.value:
            beam_track_flag_col[track_indices] = 0  # Beam track
        else:
            beam_track_flag_col[track_indices] = 1  # Scattered track

            # Only apply additional filters to scattered tracks
            # Determine side relative to beam zone for scattered tracks (Column 11) above = 1, below = -1
            side_flag_col[track_indices] = 1 if mean_y > VolumeBoundaries.BEAM_ZONE_MAX.value else -1
            side_key = 'above' if mean_y > VolumeBoundaries.BEAM_ZONE_MAX.value else 'below'

            # Calculate start and end points to check if within volume (Column 9)
            end_point, start_point, beam_vector, _, _, _ = get_directions(cluster_data[:, :3])
            within_volume = lambda point: all(VolumeBoundaries.VOLUME_MIN.value <= coord <= VolumeBoundaries.VOLUME_MAX.value for coord in point)
            volume_check_flag_col[track_indices] = 1 if (within_volume(start_point) and within_volume(end_point)) else 0

            # Check intersection point with the beam vector for volume (Column 10)
            track_vector = end_point - start_point
            intersection_point = closest_point_on_line1(start_point, track_vector, np.array([0, 128, 128]), beam_vector)
            intersection_flag_col[track_indices] = 1 if within_volume(intersection_point) else 0

            is_not_within_beam_zone = end_point[1] > VolumeBoundaries.BEAM_ZONE_MAX.value or end_point[1] < VolumeBoundaries.BEAM_ZONE_MIN.value
            if is_not_within_beam_zone:
                end_point_flag_col[track_indices] =1

            # Determine proximity flags (Column 12) for closest track to beam zone on each side

            side_tracks['above'].append((cluster_data, track_indices)) if mean_y > VolumeBoundaries.BEAM_ZONE_MAX.value else side_tracks['below'].append((cluster_data, track_indices))
    for side, tracks in side_tracks.items():
        if tracks:
            # Find the closest track to the beam zone on this side
            closest_track_data, closest_indices = min(
                [(track_data, indices) for track_data, indices in tracks],
                key=lambda t: abs(get_directions(t[0][:, :3])[1][1] - (VolumeBoundaries.BEAM_ZONE_MIN.value if side == 'below' else VolumeBoundaries.BEAM_ZONE_MAX.value)),
                default=(None, None)
            )
            if closest_track_data is not None:
                # Set proximity flag for the closest track
                proximity_flag_col[closest_indices] = 1 if side == 'above' else -1
            # Set flags for other tracks on this side
            for track_data, indices in tracks:
                if not np.array_equal(track_data, closest_track_data):
                    proximity_flag_col[indices] = 2 if side == 'above' else -2

    # Append new columns to the original data array
    data_with_flags = np.hstack([data, beam_track_flag_col, volume_check_flag_col, intersection_flag_col, side_flag_col, proximity_flag_col, end_point_flag_col])

    return data_with_flags

# Function to add plot lines
def plot_lines(track_mean, dirVecTrackNorm, start_point, end_point, intersection_point, closest_points, ax11, ax12, ax13, color = 'blue', s=400):
    line_length = 100  # Adjust line length as needed
    line_start = track_mean - line_length * dirVecTrackNorm
    line_end = track_mean + line_length * dirVecTrackNorm
    ax11.scatter(start_point[0], start_point[1], color=color, marker='x', label='Start Point', s=s)
    ax11.scatter(end_point[0], end_point[1], color=color, marker='x', label='End Point', s=int(s/2))
    # ax11.set_xlim(start_point[0]-10, end_point[0]+10)
    # ax11.set_ylim(start_point[1]-10, end_point[1]+10)
    ax12.scatter(start_point[1], start_point[2], color=color, marker='x', label='Start Point', s=s)
    ax12.scatter(end_point[1], end_point[2], color=color, marker='x', label='End Point', s=int(s/2))
    # ax12.set_xlim(start_point[1]-10, end_point[1]+10)
    # ax12.set_ylim(start_point[2]-10, end_point[2]+10)
    ax13.scatter(start_point[0], start_point[2], color=color, marker='x', label='Start Point', s=s)
    ax13.scatter(end_point[0], end_point[2], color=color, marker='x', label='End Point', s=int(s/2))
    # ax13.set_xlim(start_point[0]-10, end_point[0]+10)
    # ax13.set_ylim(start_point[2]-10, end_point[2]+10)
    ax11.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], label=f'Fitted Line')
    ax11.scatter(intersection_point[0], intersection_point[1], color=color, marker='o', label='Intersection Point', s=100)
    ax12.plot([line_start[1], line_end[1]], [line_start[2], line_end[2]], label=f'Fitted Line')
    ax12.scatter(intersection_point[1], intersection_point[2], color=color, marker='o', label='Intersection Point', s=75)
    ax13.plot([line_start[0], line_end[0]], [line_start[2], line_end[2]], label=f'Fitted Line')
    ax13.scatter(intersection_point[0], intersection_point[2], color=color, marker='o', label='Intersection Point', s=50)
    ax11.scatter(closest_points[:, 0], closest_points[:, 1], color='red', label='Closest Points on PCA Line', s=20)
    ax12.scatter(closest_points[:, 1], closest_points[:, 2], color='red', label='Closest Points on PCA Line', s=20)
    ax13.scatter(closest_points[:, 0], closest_points[:, 2], color='red', label='Closest Points on PCA Line', s=20)

# Function to find DBSCAN Clusters
def dbcluster(data_array, N_PROC, nn_neighbor, nn_radius, db_min_samples, sensitivity_, eps_threshold_, eps_mode_):
    """
    Perform DBSCAN clustering on a given data array with adaptive epsilon calculation.

    Parameters:
    - data_array: np.ndarray
        Input data with at least 3 columns (x, y, z).
    - N_PROC: int
        Number of processes for parallel computation.
    - nn_neighbor: int
        Number of nearest neighbors for the NearestNeighbors algorithm.
    - nn_radius: float
        Radius for the NearestNeighbors algorithm.
    - db_min_samples: int
        Minimum samples for a cluster in DBSCAN.
    - sensitivity_: float
        Sensitivity for the KneeLocator.
    - eps_threshold_: float
        Threshold below which epsilon defaults to eps_mode_.
    - eps_mode_: float
        Default epsilon value if calculated epsilon is below threshold.

    Returns:
    - labels_: np.ndarray
        Cluster labels from DBSCAN or [-1, -1] in case of failure.
    - valid_cluster: bool
        True if clustering is successful, False otherwise.
    - epsilon_: float
        The epsilon value used for DBSCAN.
    """
    valid_cluster = True
    epsilon_ = 0  # Default epsilon value
    try:
        # Extract the first three columns (x, y, z)
        extractedData = data_array[:, DataArray.X.value:DataArray.Z.value + 1]

        # Nearest neighbors setup
        neigh = NearestNeighbors(n_neighbors=nn_neighbor, radius=nn_radius)
        nbrs = neigh.fit(extractedData)
        distances, indices = nbrs.kneighbors(extractedData)
        distances = np.sort(distances, axis=0)
        dist_ = distances[:, 1]

        # KneeLocator to find the optimal epsilon
        kneedle = KneeLocator(
            x=indices[:, 0],
            y=dist_,
            S=sensitivity_,
            curve='convex',
            direction='increasing',
            interp_method='interp1d'
        )
        if kneedle.knee is None:
            raise ValueError("KneeLocator failed to identify a knee point.")

        epsilon_ = round(dist_[int(kneedle.knee)], 2)
        if epsilon_ < eps_threshold_:
            epsilon_ = eps_mode_

        # DBSCAN clustering
        model = DBSCAN(eps=epsilon_, min_samples=db_min_samples, n_jobs=N_PROC)
        labels_ = model.fit_predict(extractedData)
        return labels_, valid_cluster, epsilon_

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"Error: {e}")

    # Return defaults in case of failure
    return np.array([-1, -1]), False, epsilon_

# Function to do GMM clustering for every dbscan cluster
def hierarchical_clustering_with_responsibilities(data_array, max_components=10):
    """
    Perform DBSCAN clustering and then apply GMM clustering to each DBSCAN cluster,
    computing the responsibility array for all data points.

    Parameters:
    - data_array (np.ndarray): Input data array with at least 3 columns (x, y, z).
    - max_components (int): Maximum number of GMM components to evaluate for BIC.

    Returns:
    - final_labels (np.ndarray): Combined labels for the entire dataset after hierarchical clustering.
    - dbscan_labels (np.ndarray): Labels from the DBSCAN clustering.
    - final_responsibilities (np.ndarray): Responsibility matrix of shape (n_points, total_gmm_clusters).
    """
    # Step 1: Perform DBSCAN clustering
    dbscan_labels, valid_cluster, epsilon_ = dbcluster(
        data_array,
        SCAN.N_PROC.value,
        SCAN.NN_NEIGHBOR.value,
        SCAN.NN_RADIUS.value,
        SCAN.DB_MIN_SAMPLES.value,
        SCAN.SENSITIVITY.value,
        SCAN.EPS_THRESHOLD.value,
        SCAN.EPS_MODE.value
    )

    if not valid_cluster:
        print("DBSCAN clustering failed.")
        return np.array([-1] * len(data_array)), dbscan_labels, None

    unique_clusters = np.unique(dbscan_labels)
    num_points = len(data_array)

    final_labels = -1 * np.ones(num_points, dtype=int)
    final_responsibilities = -1 * np.ones((num_points, 0))

    current_label_offset = 0

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue

        cluster_mask = dbscan_labels == cluster_id
        cluster_data = data_array[cluster_mask]

        gmm_labels, n_comp, responsibilities = fit_gmm_with_bic(cluster_data, max_components=max_components)

        global_gmm_labels = gmm_labels + current_label_offset
        final_labels[cluster_mask] = global_gmm_labels

        new_responsibilities = -1 * np.ones((num_points, n_comp))
        new_responsibilities[cluster_mask, :] = responsibilities
        final_responsibilities = np.hstack((final_responsibilities, new_responsibilities))

        current_label_offset += n_comp

    return final_labels, current_label_offset, final_responsibilities, dbscan_labels

# Function to calculate the metrics for low energy tracks
def calculate_metric_low_energy(data):

    unique_gmm_labels = len(np.unique(data[:, DataArray.gmm_labels.value]))

    # Filter tracks based on flags
    scattered_tracks = data[:, DataArray.scattered_track.value] == 1  # Column 10: Scattered track
    within_volume = data[:, DataArray.track_inside_volume.value] == 1  # Column 11: Track within volume
    vertex_inside_volume = data[:, DataArray.vertex_inside_volume.value] == 1  # Column 12: Track vertex inside volume
    endpoint_above_beam = data[:, DataArray.end_point_above_beam_zone.value] == 1  # Column 15: Track endpoint above beam zone

    # Combine all conditions to filter tracks that meet the specified criteria
    valid_tracks = scattered_tracks & within_volume

    # Filter out the relevant data points
    filtered_data = data[valid_tracks]

    # Create two dictionaries: one for tracks above and one for tracks below
    tracks_above = {}
    tracks_below = {}

    # Iterate over unique track labels
    unique_labels = np.unique(filtered_data[:, DataArray.merge_p_val.value])  # Column 9: Labels

    for label in unique_labels:
        label_data = filtered_data[filtered_data[:, DataArray.merge_p_val.value] == label]

        # Classify the tracks based on column 13 (above or below)
        if np.all(label_data[:, DataArray.side_of_track.value] == 1):  # Track is above (Column 13 == 1)
            tracks_above[label] = label_data[:, :4]  # x, y, z, q (first four columns)
        elif np.all(label_data[:, DataArray.side_of_track.value] == -1):  # Track is below (Column 13 == -1)
            tracks_below[label] = label_data[:, :4]  # x, y, z, q (first four columns)

    # Check if tracks are present in either of the dictionaries
    if not tracks_above and not tracks_below:
        print("No valid tracks found above or below the beam")
        return {}

    metrics_low_energy = {}
    for tracks_dict in (tracks_above, tracks_below):
        labels = list(tracks_dict.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                label1 = labels[i]
                label2 = labels[j]
                track1 = tracks_dict[label1][:, :3]
                track2 = tracks_dict[label2][:, :3]
                end_point1, start_point1, beam_vector1, dirVecTrackNorm1, track_mean1, closest_points1 = get_directions(track1)
                end_point2, start_point2, beam_vector2, dirVecTrackNorm2, track_mean2, closest_points2 = get_directions(track2)
                dist1 = np.linalg.norm(end_point1 - start_point2)
                dist2 = np.linalg.norm(end_point2 - start_point1)
                # Calculate the number of points in each track
                num_points_track1 = len(track1)
                num_points_track2 = len(track2)
                metrics_low_energy[(label1, label2)] = (min(dist1, dist2), num_points_track1, num_points_track2, unique_gmm_labels)

    return metrics_low_energy


# Function to assign beam or scattered track based on y-position
def assign_beam_or_scattered(data_points, incoming_labels, beam_zone_min=VolumeBoundaries.BEAM_ZONE_MIN.value, beam_zone_max=VolumeBoundaries.BEAM_ZONE_MAX.value):
    label_count = defaultdict(lambda: {"beam": 0, "scattered": 0})

    # Count points for each label in or outside the beam zone
    for idx, (label, (x, y, z, Qvox)) in enumerate(zip(incoming_labels, data_points)):
        if beam_zone_min <= y <= beam_zone_max:
            label_count[label]["beam"] += 1
        else:
            label_count[label]["scattered"] += 1

    # Create a list of updated labels based on beam zone presence
    updated_labels = []
    for label in incoming_labels:
        # Calculate the number of points inside the beam zone for the current label
        total_points = label_count[label]["beam"] + label_count[label]["scattered"]
        beam_percentage = label_count[label]["beam"] / total_points if total_points > 0 else 0

        # If more than 50% of the points for the label are inside the beam zone, assign it as beam (1)
        if beam_percentage > 0.5:
            updated_labels.append(1)  # Beam
        else:
            updated_labels.append(2)  # Scattered track

    return updated_labels


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
            path_output = RunParameters.save_root_file.value
            root_file = root.TFile(path_output+RunParameters.tag.value+"_sim_5000_"+str(energy)+"mev_"+str(angle)+"cm_"+str(event_start)+"_"+str(event_end)+".root", "UPDATE")
            print(root_file)
            result = create_tree_and_branches("events")

        EventInfo = namedtuple('Events', ['event_id', 'verX', 'verY', 'verZ', 'dirX', 'dirY', 'dirZ', 'Eenergy', 'Elab', 'ransac', 'gmm', 'end_points'])
        EventInfoList = []
        exception_events = []

        for entries in myTree:
            try:
                if entries.data.event >= event_start and entries.data.event <= event_end:

                    print('Event -->', entries.data.event)
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
                                        gmm=None,
                                        end_points=None)

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
                        data_array, incoming_labels, colorbars = get_data_array(beam_center, entries, event_info)
                        if debug:
                            print('Data recorded in event')
                            print(data_array)
                    else:
                        data_array, incoming_labels = get_data_array(beam_center, entries, event_info)
                        if debug:
                            print('Data recorded in event')
                            print(data_array)

                    end_points_data_array = get_yz_min_max(data_array)
                    # print('End Points')
                    # print(end_points_data_array)

                    event_info = event_info._replace(end_points=end_points_data_array)

                    # Assign True Labels based on trackID from simulation
                    true_labels_sim = assign_beam_or_scattered(data_array, incoming_labels)
                    data_array = np.column_stack((data_array, incoming_labels))
                    data_array = np.column_stack((data_array, true_labels_sim))

                    # Assign True Labels
                    # data_array = [x,y,z,q,track_ID, true_labels_sim, true_labels_hard ]
                    data_array = generate_true_labels(data_array, event_info)
                    if debug:
                        print(data_array)

                    # Get Predicted Labels from RANSAC
                    # data_array = [0-x,1-y,2-z,3-q,4-track_ID, 5-true_labels_sim, 6-true_labels_hard, 7-ransac labels]
                    ransac_labels, fitted_models = find_multiple_lines_ransac(data_array, max_lines=10, residual_threshold=5.0, n_iterations=1000)
                    data_array = np.column_stack((data_array, ransac_labels))
                    ransac['components'] = len(np.unique(ransac_labels))
                    print('Number of unique ransac labels', np.unique(ransac_labels))

                    if plots:
                        colorbars_ransac = plot_ransac(data_array, event_info)

                    #Get Prediced Labels from GMM
                    # data_array = [0-x,1-y,2-z,3-q,4-track_ID, 5-true_labels_sim, 6-true_labels_hard, 7-ransac labels, 8-gmm labels, 9-dbscan labels, 10 - merge labels]
                    gmm_labels, n_comp, responsibilities, dbscan_labels = hierarchical_clustering_with_responsibilities(data_array, max_components=10)

                    data_array = np.column_stack((data_array, gmm_labels))
                    data_array = np.column_stack((data_array, dbscan_labels))
                    reg = Regularize(data_array=data_array, threshold=Optimize.P_VALUE.value, merge_type='p_value')
                    final_clusters = reg.merge_labels()
                    data_array = np.column_stack((data_array, final_clusters))
                    data_array = np.column_stack((data_array, final_clusters))

                    angles_ransac, intersections_ransac, start_point_ransac, end_point_ransac, phi_angle_ransac, ranges_initial, ranges_final = kinematics_ransac(data_array, fitted_models, False)
                    print('RANSAC angles', angles_ransac, ranges_final)
                    ransac['angles'] = angles_ransac
                    ransac['intersections'] = intersections_ransac
                    ransac['start_point'] = start_point_ransac
                    ransac['end_point'] = end_point_ransac
                    ransac['phi_angles'] = phi_angle_ransac

                    if RunParameters.optimize_alpha.value:
                        ransac['alpha_op'] = ranges_initial
                    else:
                        ransac['alpha_op'] = {}

                    ransac['range'] = ranges_final
                    # print('Printing alpha RANSAC')
                    # print(ranges_final)
                    # print(ranges_initial)

                    if RunParameters.optimize_beta.value:
                        lab_angles_beta_ransac = calculate_beta(data_array, model = DataArray.ransac_labels.value)
                        ransac['beta'] = lab_angles_beta_ransac
                    else:
                        ransac['beta'] = {}

                    # Create Filters to the tracks
                    # data_array = [0-x,1-y,2-z,3-q,4-track_ID, 5-true_labels_sim, 6-true_labels_hard, 7-ransac labels, 8-gmm labels, 9-dbscan labels, 10 - merge labels]
                    data_array = add_filters(data_array, model= int(DataArray.merge_p_val.value))
                    np.set_printoptions(threshold=np.inf)


                    scattered_condition = data_array[:, DataArray.scattered_track.value] == 1
                    scattered_above_mask = data_array[:, DataArray.side_of_track.value] == 1
                    scattered_below_mask = data_array[:, DataArray.side_of_track.value] == -1

                    scattered_above = scattered_condition & scattered_above_mask
                    scattered_below = scattered_condition & scattered_below_mask

                    if RunParameters.optimize_cdist.value:
                        metric_low_energy = calculate_metric_low_energy(data_array)
                        # print('Printing metric low energy')
                        # print(metric_low_energy)
                    else:
                        metric_low_energy = {}

                    original_data_array = data_array.copy()

                    if RunParameters.optimize_multiplicity.value:
                        thresholds_list = list(range(Optimize.C_DIST_RANGE_LOW.value, Optimize.C_DIST_RANGE_HIGH.value +1))
                        thresholds_list.append(Optimize.C_DIST.value)
                    else:
                        thresholds_list = [Optimize.C_DIST.value]

                    cdist_dict = {}
                    cdist_flag = False

                    for dist_thresholds in thresholds_list:
                        highest_label = max(final_clusters) + 1
                        data_array = original_data_array.copy()

                        data_array_above = data_array[scattered_above, :]
                        if data_array_above.size > 0:
                            reg_low_energy_above = Regularize(data_array=data_array_above, low_energy_threshold=dist_thresholds, merge_type='cdist', func=get_directions)
                            final_clusters_above = reg_low_energy_above.merge_labels()
                            final_clusters_above += highest_label
                            highest_label = max(final_clusters_above) + 1
                            data_array[scattered_above, DataArray.merge_cdist.value] = final_clusters_above

                        data_array_below = data_array[scattered_below, :]
                        if data_array_below.size > 0:
                            reg_low_energy_below = Regularize(data_array=data_array_below, low_energy_threshold=dist_thresholds, merge_type='cdist', func=get_directions)
                            final_clusters_below = reg_low_energy_below.merge_labels()
                            final_clusters_below += highest_label
                            data_array[scattered_below, DataArray.merge_cdist.value] = final_clusters_below
                        unique_labels_cdist = np.unique(data_array[:, DataArray.merge_cdist.value])
                        filtered_data = data_array[data_array[:, DataArray.scattered_track.value] == 1]
                        unique_labels_g, counts_g = np.unique(filtered_data[:, DataArray.merge_cdist.value], return_counts=True)
                        valid_labels = unique_labels_g[counts_g > 15]
                        cdist_dict[dist_thresholds] = (len(valid_labels), len(unique_labels_cdist))
                        if dist_thresholds == 1:
                            if len(unique_labels_cdist) == 3 and len(valid_labels) == 2:
                                cdist_flag = True

                    if RunParameters.optimize_multiplicity.value:
                        print('Multiplicity Distances')
                        print(cdist_dict)
                        # np.save('data_array.npy',data_array)

                    gmm['components'] = n_comp

                    if plots:
                        colorbars_gmm = plot_gmm(data_array, event_info)

                    angles_gmm, intersections_gmm, angles_minimize_gmm, start_point_gmm, end_point_gmm, closest_resp, closest_angle, phi_angle_gmm, data_with_filters, gmm_ranges_initial, gmm_ranges_final = kinematics_gmm(data_array, responsibilities, event_info)

                    print('GMM angles', angles_gmm, gmm_ranges_final, event_info.Elab)
                    gmm['angles'] = angles_gmm
                    gmm['intersections'] = intersections_gmm
                    gmm['start_point'] = start_point_gmm
                    gmm['end_point'] = end_point_gmm
                    gmm['resp'] = angles_minimize_gmm
                    gmm['min_res'] = closest_resp
                    gmm['min_angle'] = closest_angle
                    gmm['phi_angles'] = phi_angle_gmm
                    if RunParameters.optimize_alpha.value:
                        gmm['alpha_op'] = gmm_ranges_initial
                        print(gmm_ranges_initial)
                    else:
                        gmm['alpha_op'] = {}

                    gmm['range'] = gmm_ranges_final
                    gmm['track_dist_metric'] = metric_low_energy
                    gmm['cdist_thresholds'] = cdist_dict

                    # print('Printing alpha GMM')
                    # print(gmm_ranges_final)
                    # print(gmm_ranges_initial)

                    if RunParameters.optimize_beta.value:
                        lab_angles_beta_gmm = calculate_beta(data_array, model = DataArray.merge_cdist.value)
                        gmm['beta'] = lab_angles_beta_gmm
                    else:
                        gmm['beta'] = {}

                    # Function to calculate p values
                    if RunParameters.optimize_pij.value:
                        beam_metrics, track_metrics, beam_track_metrics = calculate_cluster_metrics(data_array, VolumeBoundaries.BEAM_ZONE_MIN.value, VolumeBoundaries.BEAM_ZONE_MAX.value)
                        # print('Metrics')
                        # print(beam_metrics)
                        # print(track_metrics)
                        # print(beam_track_metrics)
                    else:
                        beam_metrics = {}
                        track_metrics = {}
                        beam_track_metrics = {}

                    gmm['beam_beam_metric'] = beam_metrics
                    gmm['track_track_metric'] = track_metrics
                    gmm['beam_track_metric'] = beam_track_metrics

                    # Calculate the number of components beam/track
                    unique_beam_ransac, unique_track_ransac, unique_beam_gmm, unique_track_gmm = beam_track_data(data_array)
                    ransac['beam_components'] = unique_beam_ransac
                    ransac['track_components'] = unique_track_ransac
                    gmm['beam_components'] = unique_beam_gmm
                    gmm['track_components'] = unique_track_gmm


                    unique_values_ransac_reduced_k, counts_ransac_reduced_k = np.unique(data_array[:, DataArray.ransac_labels.value], return_counts=True)
                    ransac_reduced_k = dict(zip(unique_values_ransac_reduced_k, counts_ransac_reduced_k))
                    unique_values_gmm_reduced_k, counts_gmm_reduced_k = np.unique(data_array[:, DataArray.gmm_labels.value], return_counts=True)
                    gmm_reduced_k = dict(zip(unique_values_gmm_reduced_k, counts_gmm_reduced_k))
                    unique_values_p_value_reduced_k, counts_p_value_reduced_k = np.unique(data_array[:, DataArray.merge_p_val.value], return_counts=True)
                    p_value_reduced_k = dict(zip(unique_values_p_value_reduced_k, counts_p_value_reduced_k))
                    unique_values_merge_cdist_reduced_k, counts_merge_cdist_reduced_k = np.unique(data_array[:, DataArray.merge_cdist.value], return_counts=True)
                    merge_cdist_reduced_k = dict(zip(unique_values_merge_cdist_reduced_k, counts_merge_cdist_reduced_k))

                    # print('UNIQUE')
                    # print(ransac_reduced_k, gmm_reduced_k, p_value_reduced_k, merge_cdist_reduced_k)
                    # Boolean condition: unique value is not 20 and its count > 10
                    # mask = (unique_values_ransac_reduced_k != 20) & (counts_ransac_reduced_k > 10)

                    # # Total number satisfying the condition
                    # num_not_20 = np.sum(mask)

                    # if num_not_20 > 2 and cdist_flag:
                    #     print(f"Number of unique values != 20 with count > 10: {num_not_20}")
                    #     print(cdist_dict)

                    ransac_noise_mask = data_array[:, DataArray.ransac_labels.value] != 20  # Exclude points with label 20
                    gmm_noise_mask = data_array[:, DataArray.gmm_labels.value] != -1  # Exclude points with -1

                    ransac['ari'] = round(adjusted_rand_score(data_array[:, DataArray.true_labels_sim.value], data_array[:, DataArray.ransac_labels.value]), 2)
                    ransac['filtered_ari'] = round(adjusted_rand_score(data_array[ransac_noise_mask, DataArray.true_labels_sim.value], data_array[ransac_noise_mask, DataArray.ransac_labels.value]), 2)
                    ransac['label_info'] = ransac_reduced_k
                    gmm['ari'] = round(adjusted_rand_score(data_array[:, DataArray.true_labels_sim.value], data_array[:, DataArray.gmm_labels.value]), 2)
                    gmm['filtered_ari'] = round(adjusted_rand_score(data_array[gmm_noise_mask, DataArray.true_labels_sim.value], data_array[gmm_noise_mask, DataArray.gmm_labels.value]), 2)
                    gmm['label_info'] = gmm_reduced_k
                    gmm['ari_pval'] = round(adjusted_rand_score(data_array[:, DataArray.true_labels_sim.value], data_array[:, DataArray.merge_p_val.value]), 2)
                    gmm['filtered_ari_pval'] = round(adjusted_rand_score(data_array[gmm_noise_mask, DataArray.true_labels_sim.value], data_array[gmm_noise_mask, DataArray.merge_p_val.value]), 2)
                    gmm['label_info_pval'] = p_value_reduced_k
                    gmm['ari_cdist'] = round(adjusted_rand_score(data_array[:, DataArray.true_labels_sim.value], data_array[:, DataArray.merge_cdist.value]), 2)
                    gmm['filtered_ari_cdist'] = round(adjusted_rand_score(data_array[gmm_noise_mask, DataArray.true_labels_sim.value], data_array[gmm_noise_mask, DataArray.merge_cdist.value]), 2)
                    gmm['label_info_cdist'] = merge_cdist_reduced_k


                    # Append to the list of named tuples
                    event_info = event_info._replace(ransac=ransac)
                    event_info = event_info._replace(gmm=gmm)

                    EventInfoList.append(event_info)
                    # print(event_info)
                    if save_to_root:
                        # print(event_info)
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
                            for colorbar in chain(colorbars_ransac, colorbars_gmm):
                                colorbar.remove()
                            continue
                else:
                    if entries.data.event < event_start:
                        continue
                    if entries.data.event > event_end:
                        break
            except Exception as e:
                print("Exception Encountered", e)
                exception_events.append(entries.data.event)
                continue
        print('Exited the file', exception_events)
        if save_to_root:
            print('Saving to ROOT File')
            result["tree"].Write()
            root_file.Close()
            exceptions_output_path = RunParameters.exc_file_name.value
            exceptions_output_file = exceptions_output_path+RunParameters.tag.value+"_sim_5000_"+str(energy)+"mev_"+str(angle)+"cm_"+str(event_start)+"_"+str(event_end)+".npy"
            print(exceptions_output_file)
            np.save(exceptions_output_file, np.array(exception_events))
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