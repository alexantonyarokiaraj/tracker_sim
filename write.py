import ROOT
from collections import namedtuple


# Function to create the ROOT file, tree, and branches
def create_tree_and_branches(tree_name):
    # Create a TTree named "events"
    tree = ROOT.TTree(tree_name, "Event Data Tree")

    # Declare variables
    eventid = ROOT.std.vector("int")()
    endpoints = ROOT.std.vector("float")()
    verX = ROOT.std.vector("float")()
    verY = ROOT.std.vector("float")()
    verZ = ROOT.std.vector("float")()
    dirX = ROOT.std.vector("float")()
    dirY = ROOT.std.vector("float")()
    dirZ = ROOT.std.vector("float")()
    Eenergy = ROOT.std.vector("float")()
    Elab = ROOT.std.vector("float")()
    ransac_comp = ROOT.std.vector("int")()
    ransac_beam_comp = ROOT.std.vector("int")()
    ransac_track_comp = ROOT.std.vector("int")()
    ransac_ari = ROOT.std.vector("float")()
    ransac_filtered_ari = ROOT.std.vector("float")()
    ransac_labels_info = ROOT.std.vector("int")()
    ransac_labels_counts = ROOT.std.vector("int")()
    ransac_angles = ROOT.std.vector("float")()
    ransac_ranges = ROOT.std.vector("float")()
    ransac_ranges_alpha_labels = ROOT.std.vector("int")()
    ransac_ranges_alpha_counts = ROOT.std.vector("int")()
    ransac_ranges_alpha = ROOT.std.vector("float")()
    ransac_ranges_initial = ROOT.std.vector("float")()
    ransac_phi_angles = ROOT.std.vector("float")()
    ransac_inter = ROOT.std.vector("float")()
    ransac_start = ROOT.std.vector("float")()
    ransac_end = ROOT.std.vector("float")()
    ransac_vertex_dx = ROOT.std.vector("float")()
    ransac_vertex_dist3d = ROOT.std.vector("float")()
    gmm_comp = ROOT.std.vector("int")()
    gmm_beam_comp = ROOT.std.vector("int")()
    gmm_track_comp = ROOT.std.vector("int")()
    gmm_ari = ROOT.std.vector("float")()
    gmm_filtered_ari = ROOT.std.vector("float")()
    gmm_labels_info = ROOT.std.vector("int")()
    gmm_labels_counts = ROOT.std.vector("int")()
    gmm_ari_pval = ROOT.std.vector("float")()
    gmm_filtered_ari_pval = ROOT.std.vector("float")()
    p_value_labels_info = ROOT.std.vector("int")()
    p_value_labels_counts = ROOT.std.vector("int")()
    gmm_ari_cdist = ROOT.std.vector("float")()
    gmm_filtered_ari_cdist = ROOT.std.vector("float")()
    cdist_labels_info = ROOT.std.vector("int")()
    cdist_labels_counts = ROOT.std.vector("int")()
    cdist_threshold_value = ROOT.std.vector("float")()
    cdist_threshold_scattered = ROOT.std.vector("float")()
    cdist_threshold_total = ROOT.std.vector("float")()
    gmm_angles = ROOT.std.vector("float")()
    gmm_ranges = ROOT.std.vector("float")()
    gmm_ranges_alpha_counts = ROOT.std.vector("int")()
    gmm_ranges_alpha_labels = ROOT.std.vector("int")()
    gmm_ranges_alpha = ROOT.std.vector("float")()
    gmm_ranges_initial = ROOT.std.vector("float")()
    gmm_phi_angles = ROOT.std.vector("float")()
    gmm_inter = ROOT.std.vector("float")()
    gmm_start = ROOT.std.vector("float")()
    gmm_end = ROOT.std.vector("float")()
    gmm_vertex_dx = ROOT.std.vector("float")()
    gmm_vertex_dist3d = ROOT.std.vector("float")()
    gmm_resp = ROOT.std.vector("float")()
    gmm_resp_angle = ROOT.std.vector("float")()
    gmm_min_res = ROOT.std.vector("float")()
    gmm_min_angle = ROOT.std.vector("float")()
    gmm_bb_metric = ROOT.std.vector("float")()
    gmm_bb_size1 = ROOT.std.vector("int")()
    gmm_bb_size2 = ROOT.std.vector("int")()
    gmm_bb_unique_label = ROOT.std.vector("int")()
    gmm_bt_metric = ROOT.std.vector("float")()
    gmm_bt_size1 = ROOT.std.vector("int")()
    gmm_bt_size2 = ROOT.std.vector("int")()
    gmm_bt_unique_label = ROOT.std.vector("int")()
    gmm_tt_metric = ROOT.std.vector("float")()
    gmm_tt_size1 = ROOT.std.vector("int")()
    gmm_tt_size2 = ROOT.std.vector("int")()
    gmm_tt_unique_label = ROOT.std.vector("int")()
    gmm_td_metric = ROOT.std.vector("float")()
    gmm_td_size1 = ROOT.std.vector("int")()
    gmm_td_size2 = ROOT.std.vector("int")()
    gmm_td_unique_label = ROOT.std.vector("int")()

    beta_ransac_tracks = ROOT.std.vector("float")()
    beta_ransac_counts = ROOT.std.vector("float")()
    beta_ransac = ROOT.std.vector("float")()
    beta_ransac_angle = ROOT.std.vector("float")()
    beta_gmm_tracks = ROOT.std.vector("float")()
    beta_gmm_counts = ROOT.std.vector("float")()
    beta_gmm = ROOT.std.vector("float")()
    beta_gmm_angle = ROOT.std.vector("float")()

    # Create branches for each variable
    tree.Branch("eventid", eventid)
    tree.Branch("endpoints", endpoints)
    tree.Branch("verX", verX)
    tree.Branch("verY", verY)
    tree.Branch("verZ", verZ)
    tree.Branch("dirX", dirX)
    tree.Branch("dirY", dirY)
    tree.Branch("dirZ", dirZ)
    tree.Branch("Eenergy", Eenergy)
    tree.Branch("Elab", Elab)
    tree.Branch("ransac_comp", ransac_comp)
    tree.Branch("ransac_beam_comp", ransac_beam_comp)
    tree.Branch("ransac_track_comp", ransac_track_comp)
    tree.Branch("ransac_ari", ransac_ari)
    tree.Branch("ransac_ari", ransac_ari)
    tree.Branch("ransac_filtered_ari", ransac_filtered_ari)
    tree.Branch("ransac_labels_info", ransac_labels_info)
    tree.Branch("ransac_labels_counts", ransac_labels_counts)
    tree.Branch("ransac_angles", ransac_angles)
    tree.Branch("ransac_ranges", ransac_ranges)
    tree.Branch("ransac_ranges_alpha_labels", ransac_ranges_alpha_labels)
    tree.Branch("ransac_ranges_alpha_counts", ransac_ranges_alpha_counts)
    tree.Branch("ransac_ranges_alpha", ransac_ranges_alpha)
    tree.Branch("ransac_ranges_initial", ransac_ranges_initial)
    tree.Branch("ransac_phi_angles", ransac_phi_angles)
    tree.Branch("ransac_inter", ransac_inter)
    tree.Branch("ransac_start", ransac_start)
    tree.Branch("ransac_end", ransac_end)
    tree.Branch("ransac_vertex_dx", ransac_vertex_dx)
    tree.Branch("ransac_vertex_dist3d", ransac_vertex_dist3d)
    tree.Branch("gmm_comp", gmm_comp)
    tree.Branch("gmm_beam_comp", gmm_beam_comp)
    tree.Branch("gmm_track_comp", gmm_track_comp)
    tree.Branch("gmm_ari", gmm_ari)
    tree.Branch("gmm_filtered_ari", gmm_filtered_ari)
    tree.Branch("gmm_labels_info", gmm_labels_info)
    tree.Branch("gmm_labels_counts", gmm_labels_counts)
    tree.Branch("gmm_ari_pval", gmm_ari_pval)
    tree.Branch("gmm_filtered_ari_pval", gmm_filtered_ari_pval)
    tree.Branch("p_value_labels_info", p_value_labels_info)
    tree.Branch("p_value_labels_counts", p_value_labels_counts)
    tree.Branch("gmm_ari_cdist", gmm_ari_cdist)
    tree.Branch("gmm_filtered_ari_cdist", gmm_filtered_ari_cdist)
    tree.Branch("cdist_labels_info", cdist_labels_info)
    tree.Branch("cdist_labels_counts", cdist_labels_counts)
    tree.Branch("cdist_threshold_value", cdist_threshold_value)
    tree.Branch("cdist_threshold_scattered", cdist_threshold_scattered)
    tree.Branch("cdist_threshold_total", cdist_threshold_total)
    tree.Branch("gmm_angles", gmm_angles)
    tree.Branch("gmm_ranges", gmm_ranges)
    tree.Branch("gmm_ranges_alpha_counts", gmm_ranges_alpha_counts)
    tree.Branch("gmm_ranges_alpha_labels", gmm_ranges_alpha_labels)
    tree.Branch("gmm_ranges_alpha", gmm_ranges_alpha)
    tree.Branch("gmm_ranges_initial", gmm_ranges_initial)
    tree.Branch("gmm_phi_angles", gmm_phi_angles)
    tree.Branch("gmm_inter", gmm_inter)
    tree.Branch("gmm_start", gmm_start)
    tree.Branch("gmm_end", gmm_end)
    tree.Branch("gmm_vertex_dx", gmm_vertex_dx)
    tree.Branch("gmm_vertex_dist3d", gmm_vertex_dist3d)
    tree.Branch("gmm_resp", gmm_resp)
    tree.Branch("gmm_resp_angle", gmm_resp_angle)
    tree.Branch("gmm_min_res", gmm_min_res)
    tree.Branch("gmm_min_angle", gmm_min_angle)
    tree.Branch("gmm_bb_metric", gmm_bb_metric)
    tree.Branch("gmm_bb_size1", gmm_bb_size1)
    tree.Branch("gmm_bb_size2", gmm_bb_size2)
    tree.Branch("gmm_bb_unique_label", gmm_bb_unique_label)
    tree.Branch("gmm_bt_metric", gmm_bt_metric)
    tree.Branch("gmm_bt_size1", gmm_bt_size1)
    tree.Branch("gmm_bt_size2", gmm_bt_size2)
    tree.Branch("gmm_bt_unique_label", gmm_bt_unique_label)
    tree.Branch("gmm_tt_metric", gmm_tt_metric)
    tree.Branch("gmm_tt_size1", gmm_tt_size1)
    tree.Branch("gmm_tt_size2", gmm_tt_size2)
    tree.Branch("gmm_tt_unique_label", gmm_tt_unique_label)
    tree.Branch("gmm_td_metric", gmm_td_metric)
    tree.Branch("gmm_td_size1", gmm_td_size1)
    tree.Branch("gmm_td_size2", gmm_td_size2)
    tree.Branch("gmm_td_unique_label", gmm_td_unique_label)
    tree.Branch("beta_ransac_tracks", beta_ransac_tracks)
    tree.Branch("beta_ransac_counts", beta_ransac_counts)
    tree.Branch("beta_ransac", beta_ransac)
    tree.Branch("beta_ransac_angle", beta_ransac_angle)
    tree.Branch("beta_gmm_tracks", beta_gmm_tracks)
    tree.Branch("beta_gmm_counts", beta_gmm_counts)
    tree.Branch("beta_gmm", beta_gmm)
    tree.Branch("beta_gmm_angle", beta_gmm_angle)

    return {
        "tree": tree,
        "eventid": eventid,
        "endpoints": endpoints,
        "verX": verX,
        "verY": verY,
        "verZ": verZ,
        "dirX": dirX,
        "dirY": dirY,
        "dirZ": dirZ,
        "Eenergy": Eenergy,
        "Elab": Elab,
        "ransac_comp": ransac_comp,
        "ransac_beam_comp": ransac_beam_comp,
        "ransac_track_comp": ransac_track_comp,
        "ransac_ari": ransac_ari,
        "ransac_filtered_ari": ransac_filtered_ari,
        "ransac_labels_info": ransac_labels_info,
        "ransac_labels_counts": ransac_labels_counts,
        "ransac_angles": ransac_angles,
        "ransac_ranges": ransac_ranges,
        "ransac_ranges_alpha_labels": ransac_ranges_alpha_labels,
        "ransac_ranges_alpha_counts": ransac_ranges_alpha_counts,
        "ransac_ranges_alpha": ransac_ranges_alpha,
        "ransac_ranges_initial": ransac_ranges_initial,
        "ransac_phi_angles": ransac_phi_angles,
        "ransac_inter": ransac_inter,
        "ransac_start": ransac_start,
        "ransac_end": ransac_end,
        "ransac_vertex_dx": ransac_vertex_dx,
        "ransac_vertex_dist3d": ransac_vertex_dist3d,
        "gmm_comp": gmm_comp,
        "gmm_beam_comp": gmm_beam_comp,
        "gmm_track_comp": gmm_track_comp,
        "gmm_ari": gmm_ari,
        "gmm_filtered_ari": gmm_filtered_ari,
        "gmm_labels_info": gmm_labels_info,
        "gmm_labels_counts": gmm_labels_counts,
        "gmm_ari_pval": gmm_ari_pval,
        "gmm_filtered_ari_pval": gmm_filtered_ari_pval,
        "p_value_labels_info": p_value_labels_info,
        "p_value_labels_counts": p_value_labels_counts,
        "gmm_ari_cdist": gmm_ari_cdist,
        "gmm_filtered_ari_cdist": gmm_filtered_ari_cdist,
        "cdist_labels_info": cdist_labels_info,
        "cdist_labels_counts": cdist_labels_counts,
        "cdist_threshold_value": cdist_threshold_value,
        "cdist_threshold_scattered": cdist_threshold_scattered,
        "cdist_threshold_total": cdist_threshold_total,
        "gmm_angles": gmm_angles,
        "gmm_ranges": gmm_ranges,
        "gmm_ranges_alpha_counts": gmm_ranges_alpha_counts,
        "gmm_ranges_alpha_labels": gmm_ranges_alpha_labels,
        "gmm_ranges_alpha": gmm_ranges_alpha,
        "gmm_ranges_initial": gmm_ranges_initial,
        "gmm_phi_angles": gmm_phi_angles,
        "gmm_inter": gmm_inter,
        "gmm_start": gmm_start,
        "gmm_end": gmm_end,
        "gmm_vertex_dx": gmm_vertex_dx,
        "gmm_vertex_dist3d": gmm_vertex_dist3d,
        "gmm_resp": gmm_resp,
        "gmm_resp_angle": gmm_resp_angle,
        "gmm_min_res": gmm_min_res,
        "gmm_min_angle": gmm_min_angle,
        "gmm_bb_metric": gmm_bb_metric,
        "gmm_bb_size1": gmm_bb_size1,
        "gmm_bb_size2": gmm_bb_size2,
        "gmm_bb_unique_label": gmm_bb_unique_label,
        "gmm_tt_metric": gmm_tt_metric,
        "gmm_tt_size1": gmm_tt_size1,
        "gmm_tt_size2": gmm_tt_size2,
        "gmm_tt_unique_label": gmm_tt_unique_label,
        "gmm_bt_metric": gmm_bt_metric,
        "gmm_bt_size1": gmm_bt_size1,
        "gmm_bt_size2": gmm_bt_size2,
        "gmm_bt_unique_label": gmm_bt_unique_label,
        "gmm_td_metric": gmm_td_metric,
        "gmm_td_size1": gmm_td_size1,
        "gmm_td_size2": gmm_td_size2,
        "gmm_td_unique_label": gmm_td_unique_label,
        "beta_ransac_tracks": beta_ransac_tracks,
        "beta_ransac_counts": beta_ransac_counts,
        "beta_ransac": beta_ransac,
        "beta_ransac_angle": beta_ransac_angle,
        "beta_gmm_tracks": beta_gmm_tracks,
        "beta_gmm_counts": beta_gmm_counts,
        "beta_gmm": beta_gmm,
        "beta_gmm_angle": beta_gmm_angle
    }

# Function to fill event data into the tree
def fill_event_data_to_tree(result, event_data):

    # Clear the vectors before filling new data
    eventid = result["eventid"]
    endpoints = result["endpoints"]
    verX = result["verX"]
    verY = result["verY"]
    verZ = result["verZ"]
    dirX = result["dirX"]
    dirY = result["dirY"]
    dirZ = result["dirZ"]
    Eenergy = result["Eenergy"]
    Elab = result["Elab"]
    ransac_comp = result["ransac_comp"]
    ransac_beam_comp = result["ransac_beam_comp"]
    ransac_track_comp = result["ransac_track_comp"]
    ransac_ari = result["ransac_ari"]
    ransac_filtered_ari = result["ransac_filtered_ari"]
    ransac_labels_info = result["ransac_labels_info"]
    ransac_labels_counts = result["ransac_labels_counts"]
    ransac_angles = result["ransac_angles"]
    ransac_ranges = result["ransac_ranges"]
    ransac_ranges_alpha_labels = result["ransac_ranges_alpha_labels"]
    ransac_ranges_alpha_counts = result["ransac_ranges_alpha_counts"]
    ransac_ranges_alpha = result["ransac_ranges_alpha"]
    ransac_ranges_initial = result["ransac_ranges_initial"]
    ransac_phi_angles = result["ransac_phi_angles"]
    ransac_inter = result["ransac_inter"]
    ransac_start = result["ransac_start"]
    ransac_end = result["ransac_end"]
    ransac_vertex_dx = result["ransac_vertex_dx"]
    ransac_vertex_dist3d = result["ransac_vertex_dist3d"]
    gmm_comp = result["gmm_comp"]
    gmm_beam_comp = result["gmm_beam_comp"]
    gmm_track_comp = result["gmm_track_comp"]
    gmm_ari = result["gmm_ari"]
    gmm_filtered_ari = result["gmm_filtered_ari"]
    gmm_labels_info = result["gmm_labels_info"]
    gmm_labels_counts = result["gmm_labels_counts"]
    gmm_ari_pval = result["gmm_ari_pval"]
    gmm_filtered_ari_pval = result["gmm_filtered_ari_pval"]
    p_value_labels_info = result["p_value_labels_info"]
    p_value_labels_counts = result["p_value_labels_counts"]
    gmm_ari_cdist = result["gmm_ari_cdist"]
    gmm_filtered_ari_cdist = result["gmm_filtered_ari_cdist"]
    cdist_labels_info = result["cdist_labels_info"]
    cdist_labels_counts = result["cdist_labels_counts"]
    cdist_threshold_value = result["cdist_threshold_value"]
    cdist_threshold_scattered = result["cdist_threshold_scattered"]
    cdist_threshold_total = result["cdist_threshold_total"]
    gmm_angles = result["gmm_angles"]
    gmm_ranges = result["gmm_ranges"]
    gmm_ranges_alpha_counts = result["gmm_ranges_alpha_counts"]
    gmm_ranges_alpha_labels = result["gmm_ranges_alpha_labels"]
    gmm_ranges_alpha = result["gmm_ranges_alpha"]
    gmm_ranges_initial = result["gmm_ranges_initial"]
    gmm_phi_angles = result["gmm_phi_angles"]
    gmm_inter = result["gmm_inter"]
    gmm_start = result["gmm_start"]
    gmm_end = result["gmm_end"]
    gmm_vertex_dx = result["gmm_vertex_dx"]
    gmm_vertex_dist3d = result["gmm_vertex_dist3d"]
    gmm_resp = result["gmm_resp"]
    gmm_resp_angle = result["gmm_resp_angle"]
    gmm_min_res = result["gmm_min_res"]
    gmm_min_angle = result["gmm_min_angle"]
    gmm_bb_metric = result["gmm_bb_metric"]
    gmm_bb_size1 = result["gmm_bb_size1"]
    gmm_bb_size2 = result["gmm_bb_size2"]
    gmm_bb_unique_label = result["gmm_bb_unique_label"]
    gmm_bt_metric = result["gmm_bt_metric"]
    gmm_bt_size1 = result["gmm_bt_size1"]
    gmm_bt_size2 = result["gmm_bt_size2"]
    gmm_bt_unique_label = result["gmm_bt_unique_label"]
    gmm_tt_metric = result["gmm_tt_metric"]
    gmm_tt_size1 = result["gmm_tt_size1"]
    gmm_tt_size2 = result["gmm_tt_size2"]
    gmm_tt_unique_label = result["gmm_tt_unique_label"]
    gmm_td_metric = result["gmm_td_metric"]
    gmm_td_size1 = result["gmm_td_size1"]
    gmm_td_size2 = result["gmm_td_size2"]
    gmm_td_unique_label = result["gmm_td_unique_label"]
    beta_ransac_tracks = result["beta_ransac_tracks"]
    beta_ransac_counts = result["beta_ransac_counts"]
    beta_ransac = result["beta_ransac"]
    beta_ransac_angle = result["beta_ransac_angle"]
    beta_gmm_tracks = result["beta_gmm_tracks"]
    beta_gmm_counts = result["beta_gmm_counts"]
    beta_gmm = result["beta_gmm"]
    beta_gmm_angle = result["beta_gmm_angle"]


    eventid.clear()
    endpoints.clear()
    verX.clear()
    verY.clear()
    verZ.clear()
    dirX.clear()
    dirY.clear()
    dirZ.clear()
    Eenergy.clear()
    Elab.clear()
    ransac_comp.clear()
    ransac_beam_comp.clear()
    ransac_track_comp.clear()
    ransac_ari.clear()
    ransac_filtered_ari.clear()
    ransac_labels_info.clear()
    ransac_labels_counts.clear()
    ransac_angles.clear()
    ransac_ranges.clear()
    ransac_ranges_alpha_labels.clear()
    ransac_ranges_alpha_counts.clear()
    ransac_ranges_alpha.clear()
    ransac_ranges_initial.clear()
    ransac_phi_angles.clear()
    ransac_inter.clear()
    ransac_start.clear()
    ransac_end.clear()
    ransac_vertex_dx.clear()
    ransac_vertex_dist3d.clear()
    gmm_comp.clear()
    gmm_beam_comp.clear()
    gmm_track_comp.clear()
    gmm_ari.clear()
    gmm_filtered_ari.clear()
    gmm_labels_info.clear()
    gmm_labels_counts.clear()
    gmm_ari_pval.clear()
    gmm_filtered_ari_pval.clear()
    p_value_labels_info.clear()
    p_value_labels_counts.clear()
    gmm_ari_cdist.clear()
    gmm_filtered_ari_cdist.clear()
    cdist_labels_info.clear()
    cdist_labels_counts.clear()
    cdist_threshold_value.clear()
    cdist_threshold_scattered.clear()
    cdist_threshold_total.clear()
    gmm_angles.clear()
    gmm_ranges.clear()
    gmm_ranges_alpha_counts.clear()
    gmm_ranges_alpha_labels.clear()
    gmm_ranges_alpha.clear()
    gmm_ranges_initial.clear()
    gmm_phi_angles.clear()
    gmm_inter.clear()
    gmm_start.clear()
    gmm_end.clear()
    gmm_vertex_dx.clear()
    gmm_vertex_dist3d.clear()
    gmm_resp.clear()
    gmm_resp_angle.clear()
    gmm_min_res.clear()
    gmm_min_angle.clear()
    gmm_bb_metric.clear()
    gmm_bb_size1.clear()
    gmm_bb_size2.clear()
    gmm_bb_unique_label.clear()
    gmm_bt_metric.clear()
    gmm_bt_size1.clear()
    gmm_bt_size2.clear()
    gmm_bt_unique_label.clear()
    gmm_tt_metric.clear()
    gmm_tt_size1.clear()
    gmm_tt_size2.clear()
    gmm_tt_unique_label.clear()
    gmm_td_metric.clear()
    gmm_td_size1.clear()
    gmm_td_size2.clear()
    gmm_td_unique_label.clear()
    beta_ransac_tracks.clear()
    beta_ransac_counts.clear()
    beta_ransac.clear()
    beta_ransac_angle.clear()
    beta_gmm_tracks.clear()
    beta_gmm_counts.clear()
    beta_gmm.clear()
    beta_gmm_angle.clear()

    tree = result["tree"]

    # Fill vectors with the event data
    eventid.push_back(event_data.event_id)
    verX.push_back(event_data.verX)
    verY.push_back(event_data.verY)
    verZ.push_back(event_data.verZ)
    dirX.push_back(event_data.dirX)
    dirY.push_back(event_data.dirY)
    dirZ.push_back(event_data.dirZ)
    Eenergy.push_back(event_data.Eenergy)
    Elab.push_back(event_data.Elab)
    ransac_comp.push_back(event_data.ransac["components"])
    ransac_beam_comp.push_back(event_data.ransac["beam_components"])
    ransac_track_comp.push_back(event_data.ransac["track_components"])
    ransac_ari.push_back(event_data.ransac["ari"])
    ransac_filtered_ari.push_back(event_data.ransac["filtered_ari"])
    gmm_comp.push_back(event_data.gmm["components"])
    gmm_beam_comp.push_back(event_data.gmm["beam_components"])
    gmm_track_comp.push_back(event_data.gmm["track_components"])
    gmm_ari.push_back(event_data.gmm["ari"])  
    gmm_filtered_ari.push_back(event_data.gmm["filtered_ari"])
    gmm_ari_pval.push_back(event_data.gmm["ari_pval"])
    gmm_filtered_ari_pval.push_back(event_data.gmm["filtered_ari_pval"])
    gmm_ari_cdist.push_back(event_data.gmm["ari_cdist"])
    gmm_filtered_ari_cdist.push_back(event_data.gmm["filtered_ari_cdist"])


    # Iterate and Fill

    for endpoint in event_data.end_points.values():
        endpoints.push_back(endpoint)

    for range in event_data.ransac["range"].values():
        ransac_ranges.push_back(range)

    for angle in event_data.ransac["angles"].values():
        ransac_angles.push_back(angle)

    for phi_angle in event_data.ransac["phi_angles"].values():
        ransac_phi_angles.push_back(phi_angle)

    for inter in event_data.ransac["intersections"].values():
        for element in inter:
            ransac_inter.push_back(element)

    for start in event_data.ransac["start_point"].values():
        for element in start:
            ransac_start.push_back(element)

    for end_point in event_data.ransac["end_point"].values():
        for element in end_point:
            ransac_end.push_back(element)

    for dx in event_data.ransac["vertex_dx"].values():
        ransac_vertex_dx.push_back(dx)

    for dist3d in event_data.ransac["vertex_dist3d"].values():
        ransac_vertex_dist3d.push_back(dist3d)

    for inter in event_data.gmm["intersections"].values():
        for element in inter:
            gmm_inter.push_back(element)

    for start in event_data.gmm["start_point"].values():
        for element in start:
            gmm_start.push_back(element)

    for end_point in event_data.gmm["end_point"].values():
        for element in end_point:
            gmm_end.push_back(element)

    for dx in event_data.gmm["vertex_dx"].values():
        gmm_vertex_dx.push_back(dx)

    for dist3d in event_data.gmm["vertex_dist3d"].values():
        gmm_vertex_dist3d.push_back(dist3d)

    for resp in event_data.gmm["resp"].values():
        for thresholds, angles in resp.items():
            gmm_resp.push_back(thresholds)
            gmm_resp_angle.push_back(angles)

    for closest_threshold in event_data.gmm["min_res"].values():
        gmm_min_res.push_back(closest_threshold)

    for closest_angle in event_data.gmm["min_angle"].values():
        gmm_min_angle.push_back(closest_angle)

    for angle in event_data.gmm["angles"].values():
        gmm_angles.push_back(angle)

    for range in event_data.gmm["range"].values():
        gmm_ranges.push_back(range)

    for phi_angle in event_data.gmm["phi_angles"].values():
        gmm_phi_angles.push_back(phi_angle)

    for metric in event_data.gmm["beam_beam_metric"].values():
        p_value, size1, size2, unique_gmm_label = metric
        gmm_bb_metric.push_back(p_value)
        gmm_bb_size1.push_back(size1)
        gmm_bb_size2.push_back(size2)
        gmm_bb_unique_label.push_back(unique_gmm_label)

    for metric in event_data.gmm["track_track_metric"].values():
        p_value, size1, size2, unique_gmm_label = metric        
        gmm_tt_metric.push_back(p_value)
        gmm_tt_size1.push_back(int(size1))
        gmm_tt_size2.push_back(int(size2))
        gmm_tt_unique_label.push_back(int(unique_gmm_label))

    for metric in event_data.gmm["beam_track_metric"].values():
        p_value, size1, size2, unique_gmm_label = metric
        gmm_bt_metric.push_back(p_value)
        gmm_bt_size1.push_back(int(size1))
        gmm_bt_size2.push_back(int(size2))
        gmm_bt_unique_label.push_back(int(unique_gmm_label))

    for metric in event_data.gmm["track_dist_metric"].values():
        p_value, size1, size2, unique_gmm_label = metric
        gmm_td_metric.push_back(p_value)
        gmm_td_size1.push_back(int(size1))
        gmm_td_size2.push_back(int(size2))
        gmm_td_unique_label.push_back(int(unique_gmm_label))

    # Loop through the outer dictionary
    for key1, sub_dict in event_data.ransac["beta"].items():
        beta_ransac_tracks.push_back(key1)
        beta_ransac_counts.push_back(len(sub_dict))

        # Loop through the inner dictionary
        for key2, value in sub_dict.items():
            beta_ransac.push_back(key2)   # Store inner dictionary keys
            beta_ransac_angle.push_back(value)  # Store inner dictionary values

    # Loop through the outer dictionary
    for key1, sub_dict in event_data.gmm["beta"].items():
        beta_gmm_tracks.push_back(key1)
        beta_gmm_counts.push_back(len(sub_dict))

        # Loop through the inner dictionary
        for key2, value in sub_dict.items():
            beta_gmm.push_back(key2)   # Store inner dictionary keys
            beta_gmm_angle.push_back(value)  # Store inner dictionary values

    if event_data.ransac["alpha_op"]:
        # Loop through the outer dictionary
        for key1, sub_dict in event_data.ransac["alpha_op"].items():
            ransac_ranges_alpha_labels.push_back(int(key1))
            ransac_ranges_alpha_counts.push_back(len(sub_dict))
            # Loop through the inner dictionary
            for key2, value in sub_dict.items():
                ransac_ranges_alpha.push_back(key2)   # Store inner dictionary keys
                ransac_ranges_initial.push_back(value[0])  # Store inner dictionary values

    if event_data.gmm["alpha_op"]:
        # Loop through the outer dictionary
        for key1, sub_dict in event_data.gmm["alpha_op"].items():
            gmm_ranges_alpha_labels.push_back(int(key1))
            gmm_ranges_alpha_counts.push_back(len(sub_dict))

            # Loop through the inner dictionary
            for key2, value in sub_dict.items():
                gmm_ranges_alpha.push_back(key2)   # Store inner dictionary keys
                gmm_ranges_initial.push_back(value[0])  # Store inner dictionary values

    for labels, counts in event_data.ransac['label_info'].items():
            ransac_labels_info.push_back(int(labels))
            ransac_labels_counts.push_back(int(counts))

    for labels, counts in event_data.gmm['label_info'].items():
            gmm_labels_info.push_back(int(labels))
            gmm_labels_counts.push_back(int(counts))

    for labels, counts in event_data.gmm['label_info_pval'].items():
            p_value_labels_info.push_back(int(labels))
            p_value_labels_counts.push_back(int(counts))

    for labels, counts in event_data.gmm['label_info_cdist'].items():
            cdist_labels_info.push_back(int(labels))
            cdist_labels_counts.push_back(int(counts))

    for threshold, values in event_data.gmm['cdist_thresholds'].items():
        cdist_threshold_value.push_back(threshold)
        for i, value in enumerate(values):
            if i==0:
                cdist_threshold_scattered.push_back(value)
            if i==1:
                cdist_threshold_total.push_back(value)

    # Fill the tree with the data
    tree.Fill()

def main():

    # Create the tree and branches, and get the vectors for eventid, verX, angles
    # Call the function once and store the result
    result = create_tree_and_branches("events")

    EventInfo = namedtuple('Events', ['event_id', 'verX', 'verY', 'verZ', 'dirX', 'dirY', 'dirZ', 'Eenergy', 'Elab', 'ransac', 'gmm'])

    single_event = EventInfo(
    event_id=1,
    verX=1.23,
    verY=4.56,
    verZ=7.89,
    dirX=0.12,
    dirY=0.34,
    dirZ=0.56,
    Eenergy=100.0,
    Elab=50.0,
    ransac={
        "components": 3,
        "ari": 0.92,
        "beam_components": 2,
        "track_components": 4,
        "angles": {101: 23.5, 102: 45.0, 103: 67.1}
    },
    gmm={
        "components": 4,
        "ari": 0.85,
        "beam_components": 3,
        "track_components": 5,
        "angles": {201: 12.5, 202: 54.0, 203: 89.6},
        "beam_beam_metric": {(1, 2): 0.15, (2, 3): 0.18},
        "track_track_metric": {(4, 5): 0.15, (6, 7): 0.10},
        "beam_track_metric": {(2, 4): 0.18, (3, 6): 0.22}
    }
    )

    # Fill event data into the tree
    fill_event_data_to_tree(result, single_event)

    single_event = EventInfo(
    event_id=2,
    verX=10.23,
    verY=40.56,
    verZ=70.89,
    dirX=0.012,
    dirY=0.304,
    dirZ=0.506,
    Eenergy=200.0,
    Elab=500.0,
    ransac={
        "components": 1,
        "ari": 0.52,
        "beam_components": 3,
        "track_components": 5,
        "angles": {11: 3.5, 12: 5.0, 13: 7.1}
    },
    gmm={
        "components": 3,
        "ari": 0.5,
        "beam_components": 2,
        "track_components": 4,
        "angles": {21: 2.5, 20: 4.0, 23: 9.6},
        "beam_beam_metric": {(1, 2): 0.145, (2, 3): 0.218},
        "track_track_metric": {(4, 5): 0.215, (6, 7): 0.210},
        "beam_track_metric": {(2, 4): 0.218, (3, 6): 0.322}
    }
    )

     # Fill event data into the tree
    fill_event_data_to_tree(result, single_event)

    # Open the ROOT file in 'UPDATE' mode
    root_file = ROOT.TFile("events.root", "UPDATE")

    # Write the tree to the ROOT file
    result["tree"].Write()

    # Close the ROOT file
    root_file.Close()

    print("ROOT file with event data created successfully!")

if __name__ == "__main__":
    main()

