import ROOT
ROOT.gROOT.SetBatch(True)
import glob
import numpy as np
import os

# Settings
excitation_energies = [10]
cm_angles = [1]
# suppression_factor = range(32)
suppression_factor = [1]
base_path = "/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/root_files_hdbscan_filtered/"
volume_min, volume_max = 10, 246
beam_zone_min, beam_zone_max = 120, 134
hist_range = (-20, 20)
n_bins = 80
vertex_dx_range = (-100, 100)
vertex_dist3d_range = (0, 200)
n_vertex_bins = 100

def is_inside_volume(point):
    return all(volume_min <= coord <= volume_max for coord in point)

def is_outside_beam_zone(y):
    return not (beam_zone_min <= y <= beam_zone_max)

def extract_1track_vector(vec):
    return [vec[0], vec[1], vec[2]]

def process_file(filepath, branch_angle, branch_phi, branch_start, branch_end, branch_inter, input_angle, event_status=None, rejection_counts=None):
    diff_angles = []
    event_ids = []

    file = ROOT.TFile.Open(filepath)
    if not file or file.IsZombie():
        print(f"Could not open {filepath}")
        return diff_angles, event_ids

    tree = file.Get("events")
    if not tree:
        print(f"No 'events' tree in {filepath}")
        return diff_angles, event_ids

    for event in tree:
        elab = getattr(event, input_angle)
        reco_angles = getattr(event, branch_angle)
        phi_angles = getattr(event, branch_phi)
        eid = getattr(event, "eventid")

        sim_angle = elab[0]

        if len(reco_angles) != 1 or len(phi_angles) != 1:
            if rejection_counts is not None:
                key = 'not_1_track_0' if len(reco_angles) == 0 else 'not_1_track_multi'
                rejection_counts[key] = rejection_counts.get(key, 0) + 1
            continue

        phi = phi_angles[0]

        if (70 <= abs(phi) <= 110):
            if rejection_counts is not None:
                rejection_counts['phi_cut'] = rejection_counts.get('phi_cut', 0) + 1
            continue  # discard based on phi angle

        start = extract_1track_vector(getattr(event, branch_start))
        end = extract_1track_vector(getattr(event, branch_end))
        inter = extract_1track_vector(getattr(event, branch_inter))

        if not (is_inside_volume(start) and is_inside_volume(end) and is_inside_volume(inter)):
            if rejection_counts is not None:
                rejection_counts['outside_volume'] = rejection_counts.get('outside_volume', 0) + 1
            continue

        if not is_outside_beam_zone(end[1]):
            if rejection_counts is not None:
                rejection_counts['inside_beam_zone'] = rejection_counts.get('inside_beam_zone', 0) + 1
            continue

        angle_diff = sim_angle - reco_angles[0]

        if hist_range[0] < angle_diff < hist_range[1]:
            diff_angles.append(angle_diff)
            event_ids.append(eid[0])
        else:
            if rejection_counts is not None:
                rejection_counts['angle_out_of_range'] = rejection_counts.get('angle_out_of_range', 0) + 1
            # if event_status is not None:
            #     if event_status.get(int(eid[0]), 0) == 0:
            #         print('REJECTED TRACK', int(eid[0]))
                # else:
                #     print('ACCEPTED TRACK', int(eid[0]))
    file.Close()
    return diff_angles, event_ids


def process_file_vertex(filepath, branch_angle, branch_phi, branch_start, branch_end, branch_inter, branch_vdx, branch_vdist, input_angle):
    """Same filter logic as process_file; returns vertex_dx and vertex_dist3d for accepted events."""
    vdx_list = []
    vdist_list = []

    file = ROOT.TFile.Open(filepath)
    if not file or file.IsZombie():
        return vdx_list, vdist_list

    tree = file.Get("events")
    if not tree:
        file.Close()
        return vdx_list, vdist_list

    for event in tree:
        reco_angles = getattr(event, branch_angle)
        phi_angles  = getattr(event, branch_phi)

        if len(reco_angles) != 1 or len(phi_angles) != 1:
            continue

        phi = phi_angles[0]
        if 70 <= abs(phi) <= 110:
            continue

        start = extract_1track_vector(getattr(event, branch_start))
        end   = extract_1track_vector(getattr(event, branch_end))
        inter = extract_1track_vector(getattr(event, branch_inter))

        if not (is_inside_volume(start) and is_inside_volume(end) and is_inside_volume(inter)):
            continue

        if not is_outside_beam_zone(end[1]):
            continue

        sim_angle = getattr(event, input_angle)[0]
        angle_diff = sim_angle - reco_angles[0]
        if not (hist_range[0] < angle_diff < hist_range[1]):
            continue

        vdx   = getattr(event, branch_vdx)
        vdist = getattr(event, branch_vdist)
        if len(vdx) >= 1 and len(vdist) >= 1:
            vdx_list.append(float(vdx[0]))
            vdist_list.append(float(vdist[0]))

    file.Close()
    return vdx_list, vdist_list


list_values = []
# Main loop
for energy in excitation_energies:
    for cm in cm_angles:
        for suppress in suppression_factor:

            # filename = f"{energy}mev_{cm}cm.npy"
            # status_array = np.load(filename)

            # event_status = {int(row[0]): int(row[1]) for row in status_array}

            ransac_diffs = []
            gmm_diffs = []
            ransac_rejections_total = {}
            gmm_rejections_total = {}
            ransac_vdx = []
            ransac_vdist = []
            gmm_vdx = []
            gmm_vdist = []

            pattern = os.path.join(base_path, f"final_sim_5000_{energy}mev_{cm}cm_*_*_1.root")
            file_list = glob.glob(pattern)

            for filepath in file_list:
                ransac_rej = {}
                gmm_rej = {}
                ransac_diff, ransac_ids = process_file(filepath, "ransac_angles", "ransac_phi_angles",
                                                    "ransac_start", "ransac_end", "ransac_inter", "Elab",
                                                    rejection_counts=ransac_rej)
                gmm_diff, gmm_ids = process_file(filepath, "gmm_angles", "gmm_phi_angles",
                                                "gmm_start", "gmm_end", "gmm_inter", "Elab",
                                                rejection_counts=gmm_rej)

                for k, v in ransac_rej.items():
                    ransac_rejections_total[k] = ransac_rejections_total.get(k, 0) + v
                for k, v in gmm_rej.items():
                    gmm_rejections_total[k] = gmm_rejections_total.get(k, 0) + v

                ransac_diffs += ransac_diff
                gmm_diffs += gmm_diff

                r_vdx, r_vdist = process_file_vertex(
                    filepath, "ransac_angles", "ransac_phi_angles",
                    "ransac_start", "ransac_end", "ransac_inter",
                    "ransac_vertex_dx", "ransac_vertex_dist3d", "Elab")
                g_vdx, g_vdist = process_file_vertex(
                    filepath, "gmm_angles", "gmm_phi_angles",
                    "gmm_start", "gmm_end", "gmm_inter",
                    "gmm_vertex_dx", "gmm_vertex_dist3d", "Elab")
                ransac_vdx   += r_vdx
                ransac_vdist += r_vdist
                gmm_vdx   += g_vdx
                gmm_vdist += g_vdist

                # Identify GMM-only and RANSAC-only events
                gmm_only_ids = set(gmm_ids) - set(ransac_ids)
                ransac_only_ids = set(ransac_ids) - set(gmm_ids)

                # Print GMM-only events
                for eventid in gmm_only_ids:
                    print(f"EventID {eventid} is present in GMM but not in RANSAC for file: {filepath}")

                # Print RANSAC-only events
                for eventid in ransac_only_ids:
                    print(f"EventID {eventid} is present in RANSAC but not in GMM for file: {filepath}")

            # Print rejection summary
            all_keys = sorted(set(list(ransac_rejections_total.keys()) + list(gmm_rejections_total.keys())))
            print(f"\n{'='*60}")
            print(f"REJECTION SUMMARY  {energy}MeV  {cm}cm")
            print(f"{'='*60}")
            print(f"{'Filter':<30} {'RANSAC':>10} {'GMM':>10} {'Diff':>10}")
            print(f"{'-'*60}")
            for k in all_keys:
                r = ransac_rejections_total.get(k, 0)
                g = gmm_rejections_total.get(k, 0)
                print(f"{k:<30} {r:>10} {g:>10} {r-g:>+10}")
            print(f"{'-'*60}")
            print(f"{'ACCEPTED':<30} {len(ransac_diffs):>10} {len(gmm_diffs):>10} {len(ransac_diffs)-len(gmm_diffs):>+10}")
            print(f"{'='*60}\n")

            # Create canvas and histograms
            canvas = ROOT.TCanvas(f"c_{energy}_{cm}", f"Energy {energy} MeV, CM {cm}°", 1200, 600)
            canvas.Divide(2, 1)

            h_ransac = ROOT.TH1F(f"ransac_hist_{energy}_{cm}", f"RANSAC Δθ - Energy {energy} MeV, CM {cm}°;Δθ (deg);Entries", n_bins, hist_range[0], hist_range[1])
            h_gmm = ROOT.TH1F(f"gmm_hist_{energy}_{cm}", f"GMM Δθ - Energy {energy} MeV, CM {cm}°;Δθ (deg);Entries", n_bins, hist_range[0], hist_range[1])

            for diff in ransac_diffs:
                h_ransac.Fill(diff)
            for diff in gmm_diffs:
                h_gmm.Fill(diff)

            def annotate_hist(hist):
                entries = hist.GetEntries()
                mean = hist.GetMean()
                stddev = hist.GetStdDev()
                hist.SetStats(0)
                label = ROOT.TLatex()
                label.SetNDC()
                label.SetTextSize(0.03)
                label.DrawLatex(0.15, 0.85, f"Entries: {int(entries)}")
                label.DrawLatex(0.15, 0.80, f"Mean: {mean:.2f}")
                label.DrawLatex(0.15, 0.75, f"Std Dev: {stddev:.2f}")

            list_values.append([energy, cm, suppress, h_ransac.GetEntries(), h_ransac.GetMean(), h_ransac.GetStdDev(), h_gmm.GetEntries(), h_gmm.GetMean(), h_gmm.GetStdDev()])

            canvas.cd(1)
            h_ransac.SetLineColor(ROOT.kBlue)
            h_ransac.Draw()
            annotate_hist(h_ransac)

            # Add label (a)
            label_a = ROOT.TLatex()
            label_a.SetNDC()
            label_a.SetTextSize(0.04)
            label_a.DrawLatex(0.1, 0.92, "(a)")

            canvas.cd(2)
            h_gmm.SetLineColor(ROOT.kRed)
            h_gmm.Draw()
            annotate_hist(h_gmm)

            # Add label (b)
            label_b = ROOT.TLatex()
            label_b.SetNDC()
            label_b.SetTextSize(0.04)
            label_b.DrawLatex(0.1, 0.92, "(b)")


            canvas.Update()
            canvas.SaveAs(f"fhis_{energy}MeV_{cm}cm_{suppress}su_min_samples.png")

            # ---- Vertex residual ROOT file (2 pads) ----
            root_out = ROOT.TFile(f"vertex_{energy}MeV_{cm}cm.root", "RECREATE")

            h_ransac_vdx   = ROOT.TH1F(f"ransac_vdx_{energy}_{cm}",
                f"Vertex X diff {energy} MeV {cm}cm;x_{{reco}} - x_{{sim}} (mm);Entries",
                n_vertex_bins, vertex_dx_range[0], vertex_dx_range[1])
            h_gmm_vdx      = ROOT.TH1F(f"gmm_vdx_{energy}_{cm}",
                f"Vertex X diff {energy} MeV {cm}cm;x_{{reco}} - x_{{sim}} (mm);Entries",
                n_vertex_bins, vertex_dx_range[0], vertex_dx_range[1])
            h_ransac_vdist = ROOT.TH1F(f"ransac_vdist_{energy}_{cm}",
                f"Vertex 3D dist {energy} MeV {cm}cm;|v_{{reco}} - v_{{sim}}| (mm);Entries",
                n_vertex_bins, vertex_dist3d_range[0], vertex_dist3d_range[1])
            h_gmm_vdist    = ROOT.TH1F(f"gmm_vdist_{energy}_{cm}",
                f"Vertex 3D dist {energy} MeV {cm}cm;|v_{{reco}} - v_{{sim}}| (mm);Entries",
                n_vertex_bins, vertex_dist3d_range[0], vertex_dist3d_range[1])

            for v in ransac_vdx:   h_ransac_vdx.Fill(v)
            for v in gmm_vdx:      h_gmm_vdx.Fill(v)
            for v in ransac_vdist: h_ransac_vdist.Fill(v)
            for v in gmm_vdist:    h_gmm_vdist.Fill(v)

            # Build normalized clones before drawing (histograms are valid here)
            norm_hists = {}
            for h in [h_ransac_vdx, h_gmm_vdx, h_ransac_vdist, h_gmm_vdist]:
                h_norm = h.Clone(h.GetName() + "_norm")
                h_norm.SetDirectory(0)  # detach from file so it survives root_out.Close()
                if h_norm.Integral() > 0:
                    h_norm.Scale(1.0 / h_norm.Integral())
                h_norm.GetYaxis().SetTitle("Normalized counts")
                norm_hists[h.GetName()] = h_norm

            h_rvdx_n   = norm_hists[f"ransac_vdx_{energy}_{cm}"]
            h_gvdx_n   = norm_hists[f"gmm_vdx_{energy}_{cm}"]
            h_rvdist_n = norm_hists[f"ransac_vdist_{energy}_{cm}"]
            h_gvdist_n = norm_hists[f"gmm_vdist_{energy}_{cm}"]

            h_rvdx_n.SetLineColor(ROOT.kBlue)
            h_gvdx_n.SetLineColor(ROOT.kRed)
            h_rvdist_n.SetLineColor(ROOT.kBlue)
            h_gvdist_n.SetLineColor(ROOT.kRed)

            c_vertex = ROOT.TCanvas(f"c_vertex_{energy}_{cm}",
                f"Vertex Residuals {energy} MeV {cm}cm", 1200, 600)
            c_vertex.Divide(2, 1)

            c_vertex.cd(1)
            h_rvdist_n.SetStats(0)
            h_gvdist_n.SetStats(0)
            h_rvdist_n.SetMaximum(max(h_rvdist_n.GetMaximum(), h_gvdist_n.GetMaximum()) * 1.1)
            h_rvdist_n.Draw("HIST")
            h_gvdist_n.Draw("HIST SAME")
            leg_vdist = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
            leg_vdist.AddEntry(h_rvdist_n, "RANSAC", "l")
            leg_vdist.AddEntry(h_gvdist_n, "GMM",    "l")
            leg_vdist.Draw()

            c_vertex.cd(2)
            h_rvdx_n.SetStats(0)
            h_gvdx_n.SetStats(0)
            h_rvdx_n.SetMaximum(max(h_rvdx_n.GetMaximum(), h_gvdx_n.GetMaximum()) * 1.1)
            h_rvdx_n.Draw("HIST")
            h_gvdx_n.Draw("HIST SAME")
            leg_vdx = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
            leg_vdx.AddEntry(h_rvdx_n, "RANSAC", "l")
            leg_vdx.AddEntry(h_gvdx_n, "GMM",    "l")
            leg_vdx.Draw()

            c_vertex.Update()
            c_vertex.Write()
            h_ransac_vdx.Write()
            h_gmm_vdx.Write()
            h_ransac_vdist.Write()
            h_gmm_vdist.Write()
            root_out.Close()

            # Save area-normalized copies in a separate ROOT file
            # Save PNG of normalized histograms
            c_norm = ROOT.TCanvas(f"c_norm_{energy}_{cm}",
                f"Normalized Vertex Residuals {energy} MeV {cm}cm", 1200, 600)
            c_norm.Divide(2, 1)

            c_norm.cd(1)
            h_rvdist_n = norm_hists[f"ransac_vdist_{energy}_{cm}"]
            h_gvdist_n = norm_hists[f"gmm_vdist_{energy}_{cm}"]
            h_rvdist_n.SetStats(0)
            h_gvdist_n.SetStats(0)
            h_rvdist_n.SetMaximum(max(h_rvdist_n.GetMaximum(), h_gvdist_n.GetMaximum()) * 1.2)
            h_rvdist_n.Draw("HIST")
            h_gvdist_n.Draw("HIST SAME")
            leg1 = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
            leg1.AddEntry(h_rvdist_n, "RANSAC", "l")
            leg1.AddEntry(h_gvdist_n, "GMM", "l")
            leg1.Draw()

            c_norm.cd(2)
            h_rvdx_n = norm_hists[f"ransac_vdx_{energy}_{cm}"]
            h_gvdx_n = norm_hists[f"gmm_vdx_{energy}_{cm}"]
            h_rvdx_n.SetStats(0)
            h_gvdx_n.SetStats(0)
            h_rvdx_n.SetMaximum(max(h_rvdx_n.GetMaximum(), h_gvdx_n.GetMaximum()) * 1.2)
            h_rvdx_n.Draw("HIST")
            h_gvdx_n.Draw("HIST SAME")
            leg2 = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
            leg2.AddEntry(h_rvdx_n, "RANSAC", "l")
            leg2.AddEntry(h_gvdx_n, "GMM", "l")
            leg2.Draw()

            c_norm.Update()
            c_norm.SaveAs(f"vertex_norm_{energy}MeV_{cm}cm.png")

            # Write normalized histograms and canvas to separate ROOT file
            root_norm = ROOT.TFile(f"vertex_norm_{energy}MeV_{cm}cm.root", "RECREATE")
            for h_norm in norm_hists.values():
                h_norm.Write()
            c_norm.Write()
            root_norm.Close()



# np.save('list_e1e2metric.npy',np.array(list_values))