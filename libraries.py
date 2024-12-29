from enum import Enum

class DataArray(Enum):
    X = 0
    Y = 1
    Z = 2
    Q = 3
    true_labels = 4
    ransac_labels = 5
    gmm_labels = 6
    dbscan_labels = 7
    merge_p_val =  8
    merge_cdist = 9
    scattered_track = 10
    track_inside_volume = 11
    vertex_inside_volume = 12
    side_of_track = 13
    closest_track = 14
    end_point_above_beam_zone = 15



