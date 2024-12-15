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
plots = False
sim = True
debug=False
final_plots_flag = False
event_start = int(split_strings[2])
event_end = int(split_strings[3])
save_final_data=False
with_missing_pads = True
batch_mode = True
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

class VolumeBoundaries(Enum):
    VOLUME_MIN = 10
    VOLUME_MAX = 246
    BEAM_ZONE_MIN = 122
    BEAM_ZONE_MAX = 132
    BEAM_CENTER = 128

class SCAN(Enum):
    N_PROC = 1
    NN_NEIGHBOR = 2
    NN_RADIUS = 20.0
    DB_MIN_SAMPLES = 4
    SENSITIVITY = 3
    EPS_THRESHOLD = 5
    EPS_MODE = 10

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

# Function to plot the kinematics of RANSAC Clusters
def kinematics_ransac(data, fitted_models, useLineModelND):
    # Define the beam line endpoints
    data = data[data[:, 5] != 20]
    data = add_filters(data, model=int(5))
    lab_angles_initial = {}
    intersections_initial = {}
    phi_angles_initial = {}
    start_point_initial = {}
    end_point_initial = {}
    ransac_labels = np.unique(data[:, 5])
    for label in ransac_labels:
        # Get the points corresponding to the current label
        cluster_data = data[data[:, 5] == label]

        # Check if the cluster size is greater than 10
        if cluster_data.shape[0] <= 10:
            continue  # Skip this cluster if it has 10 or fewer points

        # Calculate the mean y of the cluster
        mean_y = np.mean(cluster_data[:, 1])
        # Process clusters either above or below the beam line
        if (mean_y >= VolumeBoundaries.BEAM_ZONE_MAX.value or mean_y < VolumeBoundaries.BEAM_ZONE_MIN.value) and cluster_data.shape[0] >= 10:
            mask = (
                (cluster_data[:, 7] == 1) &  # Column 8: (beam_track_flag_col) = 1
                (cluster_data[:, 8] == 1) &  # Column 9: (volume_check_flag_col) = 1
                (cluster_data[:, 9] == 1) &  # Column 10: (intersection_flag_col) = 1
                ((cluster_data[:, 10] == 1) | (cluster_data[:, 10] == -1)) &  # Column 11: (side_flag_col) = 1 or -1
                ((cluster_data[:, 11] == 1) | (cluster_data[:, 11] == -1)) & # Column 12: (proximity_flag_col) = 1 or -1
                (cluster_data[:, 12] == 1) # Column 12: Track End point above the beam zone
            )
            if cluster_data[mask, :3].size > 0:
                # Fill the lab angles and intersections based on first PCA fit.
                end_point, start_point, beam_vector, dirVecTrackNorm, track_mean, closest_points = get_directions(cluster_data[mask, :3])
                track_vector = end_point - start_point
                lab_angle = angle_between(track_vector, beam_vector)
                phi_angle = calculate_phi_angle(track_vector, beam_vector)
                intersection_point = closest_point_on_line1(start_point, track_vector, np.array([0,128,128]), beam_vector)
                lab_angles_initial[label] = round(lab_angle, 2)
                intersections_initial[label] = intersection_point
                start_point_initial[label] = start_point
                end_point_initial[label] = end_point
                phi_angles_initial[label] = phi_angle
                if plots:
                    plot_lines(track_mean, dirVecTrackNorm, start_point, end_point, intersection_point, closest_points, ax8, ax9, ax10)
                    if not useLineModelND:
                        threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2 = return_threshold_lines(track_mean, dirVecTrackNorm)
                        plot_threshold_lines(track_mean, dirVecTrackNorm, threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2)
                    else:
                        model_params = fitted_models[label].params
                        threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2 = return_threshold_lines(np.array(model_params[0]), np.array(model_params[1]))
                        plot_threshold_lines(np.array(model_params[0]), np.array(model_params[1]), threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2)
                # print('Track inside Volume', start_point, end_point, intersection_point, np.unique(cluster_data[:, 10]), np.unique(cluster_data[:, 11]), np.unique(cluster_data[:, 12]))
            else:
                end_point, start_point, beam_vector, dirVecTrackNorm, track_mean, closest_points = get_directions(cluster_data[:, :3])
                track_vector = end_point - start_point
                intersection_point = closest_point_on_line1(start_point, track_vector, np.array([0,128,128]), beam_vector)
                # print('Track not inside Volume', start_point, end_point, intersection_point, np.unique(cluster_data[:, 10]), np.unique(cluster_data[:, 11]), np.unique(cluster_data[:, 12]))
                # if 0 in np.unique(cluster_data[:, 12]):
                #     print('Track End point inside')
                if plots:
                    plot_lines(track_mean, dirVecTrackNorm, start_point, end_point, intersection_point, closest_points, ax8, ax9, ax10)
                    if not useLineModelND:
                        threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2 = return_threshold_lines(track_mean, dirVecTrackNorm)
                        plot_threshold_lines(track_mean, dirVecTrackNorm, threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2)
                    else:
                        model_params = fitted_models[label].params
                        threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2 = return_threshold_lines(np.array(model_params[0]), np.array(model_params[1]))
                        plot_threshold_lines(np.array(model_params[0]), np.array(model_params[1]), threshold_line_above1, threshold_line_below1, threshold_line_above2, threshold_line_below2)
    return lab_angles_initial, intersections_initial, start_point_initial, end_point_initial, phi_angles_initial

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
def kinematics_gmm(data, responsibilities, event_info):
    """
    Analyze clusters based on GMM labels to find direction vectors, intersections with a beam line,
    and angles. Plot the clusters with their fitted line in XY projection.

    Parameters:
    - data (np.ndarray): Input data array with columns [x, y, z, q, true labels, ransac labels, gmm labels].

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

    data = add_filters(data, model= int(6))
    gmm_labels = np.unique(data[:, 6])
    for label in gmm_labels:
        cluster_data = data[data[:, 6] == label]
        if len(cluster_data) < 10:
            continue  # Skip clusters with fewer than 10 points
        # Calculate the mean of Y coordinate
        mean_y = np.mean(cluster_data[:, 1])
        if (mean_y >= VolumeBoundaries.BEAM_ZONE_MAX.value or mean_y < VolumeBoundaries.BEAM_ZONE_MIN.value) and cluster_data.shape[0] >= 10:
            mask = (
                    (cluster_data[:, 7] == 1) &  # Column 8: (beam_track_flag_col) = 1
                    (cluster_data[:, 8] == 1) &  # Column 9: (volume_check_flag_col) = 1
                    (cluster_data[:, 9] == 1) &  # Column 10: (intersection_flag_col) = 1
                    ((cluster_data[:, 10] == 1) | (cluster_data[:, 10] == -1)) &  # Column 11: (side_flag_col) = 1 or -1
                    ((cluster_data[:, 11] == 1) | (cluster_data[:, 11] == -1)) & # Column 12: (proximity_flag_col) = 1 or 2
                    (cluster_data[:, 12] == 1) # Column 12: Track End point above the beam zone
                )
            if cluster_data[mask, :3].size > 0:
                # Fill the lab angles and intersections based on first PCA fit.
                end_point, start_point, beam_vector, dirVecTrackNorm, track_mean, closest_points = get_directions(cluster_data[mask, :3])
                track_vector = end_point - start_point
                lab_angle = angle_between(track_vector, beam_vector)
                phi_angle = calculate_phi_angle(track_vector, beam_vector)
                intersection_point = closest_point_on_line1(start_point, track_vector, np.array([0,128,128]), beam_vector)
                lab_angles_initial[label] = round(lab_angle, 2)
                intersections_initial[label] = intersection_point
                start_point_initial[label] = start_point
                end_point_initial[label] = end_point
                phi_angles_initial[label] = phi_angle
                if plots:
                    plot_lines(track_mean, dirVecTrackNorm, start_point, end_point, intersection_point, closest_points, ax11, ax12, ax13)
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
            beam_zone_mask = (data_array[:, 1] >= 120) & (data_array[:, 1] <= 134)
            labels_for_current_label = data_array[:, 6] == label
            not_belonging_to_label = ~labels_for_current_label
            inside_beam_zone_not_label = beam_zone_mask & not_belonging_to_label
            res = np.concatenate([range_1, range_2, range_3, range_4, range_5, range_6, range_7])
            # res = [0.1]
            if cluster_data[mask, :3].size > 0:
                for res_threshold in res:
                    responsibility_threshold = res_threshold
                    responsibility_mask = responsibilities[:, int(label)] > responsibility_threshold
                    final_mask = inside_beam_zone_not_label & responsibility_mask
                    data_for_angle = np.vstack((cluster_data[mask, :3], data[final_mask, :3]))
                    end_point, start_point, beam_vector, dirVecTrackNorm, track_mean, closest_points = get_directions(data_for_angle)
                    track_vector = end_point - start_point
                    lab_angle_p = angle_between(track_vector, beam_vector)
                    lab_angles_resp[res_threshold] = round(lab_angle_p, 2)
                lab_angles_minimize[label] = lab_angles_resp
                closest_threshold, closest_angle = min(lab_angles_resp.items(), key=lambda item: abs(item[1] - event_info.Elab))
                closest_threshold_dict[label] = closest_threshold
                closest_angle_dict[label] = closest_angle
                responsibility_threshold = closest_threshold
                responsibility_mask = responsibilities[:, int(label)] > responsibility_threshold
                final_mask = inside_beam_zone_not_label & responsibility_mask
                data_for_angle = np.vstack((cluster_data[mask, :3], data[final_mask, :3]))
                end_point, start_point, beam_vector, dirVecTrackNorm, track_mean, closest_points = get_directions(data_for_angle)
                track_vector = end_point - start_point
                print('Lowest Angle, Threshold', round(angle_between(track_vector, beam_vector), 2), closest_threshold*100)
                if plots:
                    intersection_point = closest_point_on_line1(start_point, track_vector, np.array([0,128,128]), beam_vector)
                    plot_lines(track_mean, dirVecTrackNorm, start_point, end_point, intersection_point, closest_points, ax11, ax12, ax13)
                    ax11.scatter(data[final_mask, 0]+1, data[final_mask, 1]+1, marker = 'o')
                    ax12.scatter(data[final_mask, 1]+1, data[final_mask, 2]+1, marker = 'o')
                    ax13.scatter(data[final_mask, 0]+1, data[final_mask, 2]+1, marker = 'o')
    return lab_angles_initial, intersections_initial, lab_angles_minimize, start_point_initial, end_point_initial, closest_threshold_dict, closest_angle_dict, phi_angles_initial

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
    # - data (np.ndarray): Input data array with columns [x, y, z, q, true labels, ransac labels, gmm labels].

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
        mean_y = np.mean(cluster_data[:, 1])
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
def plot_lines(track_mean, dirVecTrackNorm, start_point, end_point, intersection_point, closest_points, ax11, ax12, ax13):
    line_length = 100  # Adjust line length as needed
    line_start = track_mean - line_length * dirVecTrackNorm
    line_end = track_mean + line_length * dirVecTrackNorm
    ax11.scatter(start_point[0], start_point[1], color='blue', marker='x', label='Start Point', s=500)
    ax11.scatter(end_point[0], end_point[1], color='green', marker='x', label='End Point', s=200)
    # ax11.set_xlim(start_point[0]-10, end_point[0]+10)
    # ax11.set_ylim(start_point[1]-10, end_point[1]+10)
    ax12.scatter(start_point[1], start_point[2], color='blue', marker='x', label='Start Point', s=500)
    ax12.scatter(end_point[1], end_point[2], color='green', marker='x', label='End Point', s=200)
    # ax12.set_xlim(start_point[1]-10, end_point[1]+10)
    # ax12.set_ylim(start_point[2]-10, end_point[2]+10)
    ax13.scatter(start_point[0], start_point[2], color='blue', marker='x', label='Start Point', s=500)
    ax13.scatter(end_point[0], end_point[2], color='green', marker='x', label='End Point', s=200)
    # ax13.set_xlim(start_point[0]-10, end_point[0]+10)
    # ax13.set_ylim(start_point[2]-10, end_point[2]+10)
    ax11.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], label=f'Fitted Line')
    ax11.scatter(intersection_point[0], intersection_point[1], color='blue', marker='o', label='Intersection Point', s=100)
    ax12.plot([line_start[1], line_end[1]], [line_start[2], line_end[2]], label=f'Fitted Line')
    ax12.scatter(intersection_point[1], intersection_point[2], color='blue', marker='o', label='Intersection Point', s=100)
    ax13.plot([line_start[0], line_end[0]], [line_start[2], line_end[2]], label=f'Fitted Line')
    ax13.scatter(intersection_point[0], intersection_point[2], color='blue', marker='o', label='Intersection Point', s=100)
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
        extractedData = data_array[:, 0:3]

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
            path_output = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/"
            root_file = root.TFile(path_output+"recon_sim_5000_"+str(energy)+"mev_"+str(angle)+"cm_"+str(event_start)+"_"+str(event_end)+".root", "UPDATE")
            print(root_file)
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

                    #Get Prediced Labels from GMM
                    # data_array = [0-x,1-y,2-z,3-q, 4-true labels, 5-ransac labels, 6-gmm labels, 7-dbscan labels]
                    gmm_labels, n_comp, responsibilities, dbscan_labels = hierarchical_clustering_with_responsibilities(data_array, max_components=10)
                    data_array = np.column_stack((data_array, gmm_labels))
                    data_array = np.column_stack((data_array, dbscan_labels))
                    gmm['components'] = n_comp

                    if plots:
                        colorbars_gmm = plot_gmm(data_array, event_info)

                    angles_ransac, intersections_ransac, start_point_ransac, end_point_ransac, phi_angle_ransac = kinematics_ransac(data_array, fitted_models, False)
                    print('RANSAC angles', angles_ransac)
                    ransac['angles'] = angles_ransac
                    ransac['intersections'] = intersections_ransac
                    ransac['start_point'] = start_point_ransac
                    ransac['end_point'] = end_point_ransac
                    ransac['phi_angles'] = phi_angle_ransac

                    angles_gmm, intersections_gmm, angles_minimize_gmm, start_point_gmm, end_point_gmm, closest_resp, closest_angle, phi_angle_gmm = kinematics_gmm(data_array, responsibilities, event_info)
                    print('GMM angles', angles_gmm)
                    gmm['angles'] = angles_gmm
                    gmm['intersections'] = intersections_gmm
                    gmm['start_point'] = start_point_gmm
                    gmm['end_point'] = end_point_gmm
                    gmm['resp'] = angles_minimize_gmm
                    gmm['min_res'] = closest_resp
                    gmm['min_angle'] = closest_angle
                    gmm['phi_angles'] = phi_angle_gmm

                    # print('GMM intersections', intersections_gmm)
                    # print('minimize', angles_minimize_gmm)

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
                            for colorbar in chain(colorbars, colorbars_ransac, colorbars_gmm):
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