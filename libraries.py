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



