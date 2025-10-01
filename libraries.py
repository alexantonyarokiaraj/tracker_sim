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
    old_ransac_labels = 18

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
    zoom_in_length = 40
    optimize_alpha = True
    optimize_beta = True
    optimize_gamma =  True
    optimize_cdist = False
    optimize_pij = True
    optimize_multiplicity = False
    save_root_file = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/root_files_1/"
    save_root_fig = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/images/compare/3cm/"
    tag = "final"
    exc_file_name = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/root_files_1/"
    range_lookup_table = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/LookupTable_e780_58Ni_68Ni_Alex.xlsx"
    range_energy_conversion_sheet = "range_energy_he_he_cf4_mixed"
    use_cij_ransac = True
    use_beta_fraction = True
    calculate_geometric_efficiency = True
    use_iterative_ransac = False

class VolumeBoundaries(Enum):
    VOLUME_MIN = 10
    VOLUME_MAX = 246
    BEAM_ZONE_MIN = 122
    BEAM_ZONE_MAX = 132
    BEAM_CENTER = 128

class SCAN(Enum):
    N_PROC = 1
    NN_NEIGHBOR = 5
    NN_RADIUS = 20.0
    DB_MIN_SAMPLES = 4
    SENSITIVITY = 3
    EPS_THRESHOLD = 4.0
    EPS_MODE = 4.0

class Optimize(Enum):
    ALPHA = 28.5/100 #percentage
    ALPHA_RANGE_LOW = 50/100
    ALPHA_RANGE_HIGH = 50/100
    ALPHA_STEPS = 1
    BETA = 40 #mm
    BETA_RANGE_LOW = 10 #mm
    BETA_RANGE_HIGH = 100 #mm
    BETA_STEPS = 1 #mm
    BETA_FRACTION = 55/100
    BETA_RANGE_LOW_FRACTION = 1/100
    BETA_RANGE_HIGH_FRACTION = 100/100
    BETA_STEPS_FRACTION = 100
    GAMMA = 1.0/100
    P_VALUE = 0.1
    C_DIST = 15
    C_DIST_RANGE_LOW = 1
    C_DIST_RANGE_HIGH = 100

class FileNames(Enum):
    CALIBRATION_PADS = 'pad_calibration_actar.txt'
    MISSING_PADS = 'HitResponses.dat'
    CONVERSION_TABLE = 'LT_GANIL_NewCF_marine.dat'
    CONFIG_FILE_EXCEL = 'LookupTable_e780_58Ni_68Ni_Alex.xlsx'
    RANGE_ENERGY_CONVERSION_SHEET = "range_energy_he_he_cf4_mixed"

class Reference(Enum):
    RANGE_EXTEND = 40
    RANGE_BIN_SIZE = 2
    RANGE_BIN_PER = 20
    AREA_TOTAL_PAD = 4
    LINE_LENGTH_THRESHOLD = Optimize.BETA.value  # Threshold to define smaller or larger tracks
    SAVITZKY_GOLAY_WINDOW_LARGE = 7  # Window Length for Savitzky Golay Filter for large tracks
    SAVITZKY_GOLAY_WINDOW_SMALL = 5  # Window Length for Savitzky Golay Filter for small tracks
    THRESHOLD_PEAKS = 0.25

class ConversionFactors(Enum):
    DRIFT_VELOCITY = 1.16  # units [cm/us]
    Z_CONVERSION_FACTOR = DRIFT_VELOCITY * (10.0 / 1000.0)  # mm/us
    X_CONVERSION_FACTOR = 2.0  # mm
    Y_CONVERSION_FACTOR = 2.0  # mm
    NBINS_X = 128
    X_START_BIN = 0 * X_CONVERSION_FACTOR
    X_END_BIN = 128 * X_CONVERSION_FACTOR
    NBINS_Y = 128
    Y_START_BIN = 0 * Y_CONVERSION_FACTOR
    Y_END_BIN = 128 * Y_CONVERSION_FACTOR
    NBINS_Z = 28000
    Z_START_BIN = 0 * Z_CONVERSION_FACTOR
    Z_END_BIN = 28000 * Z_CONVERSION_FACTOR

class RansacParameters(Enum):
    MAX_LINES = 10
    RESIDUAL_THRESHOLD = 5.0
    N_ITERATIONS = 5000