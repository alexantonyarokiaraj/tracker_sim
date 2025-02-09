from enum import Enum

class DataArray(Enum):
    X = 0
    Y = 1
    Z = 2
    Q = 3
    trackID = 4
    true_labels_sim = 5
    true_labels_hard = 6
    ransac_labels = 7
    gmm_labels = 8
    dbscan_labels = 9
    merge_p_val =  10
    merge_cdist = 11
    scattered_track = 12
    track_inside_volume = 13
    vertex_inside_volume = 14
    side_of_track = 15
    closest_track = 16
    end_point_above_beam_zone = 17

class RunParameters(Enum):
    sim = True
    plots = True
    debug=False
    final_plots_flag = False
    save_final_data=False
    with_missing_pads = True
    batch_mode = False
    save_to_root = False
    save_python_figures = False

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

class Optimize(Enum):
    ALPHA = 36.5/100
    BETA = 40

class FileNames(Enum):
    CALIBRATION_PADS = 'pad_calibration_actar.txt'
    MISSING_PADS = 'HitResponses.dat'

class Reference(Enum):
    RANGE_EXTEND = 20