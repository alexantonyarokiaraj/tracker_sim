import ROOT
import glob
import os

# --- Settings ---
excitation_energies = [10]
cm_angles = [5]
suppression_factor = [1]
base_path = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/root_files_2/"
volume_min, volume_max = 10, 246
beam_zone_min, beam_zone_max = 120, 134

required_branches = [
    "ransac_ari",
    "ransac_filtered_ari",
    "gmm_ari",
    "gmm_ari_pval",
    "gmm_ari_cdist"
]

angle_hist_range = (-20, 20)
ari_xrange = (0.0, 1.2)
n_bins = 50

# --- Helper functions ---
def is_inside_volume(point):
    return all(volume_min <= coord <= volume_max for coord in point)

def is_outside_beam_zone(y):
    return not (beam_zone_min <= y <= beam_zone_max)

def extract_1track_vector(vec):
    return [vec[0], vec[1], vec[2]]

# --- Process one file for rejected events ---
def process_file_for_rejected_aris(filepath):
    f = ROOT.TFile.Open(filepath)
    if not f or f.IsZombie():
        print(f"⚠️ Could not open {filepath}")
        return {b: [] for b in required_branches}
    tree = f.Get("events")
    if not tree:
        f.Close()
        return {b: [] for b in required_branches}

    ari_data = {b: [] for b in required_branches}

    for event in tree:
        try:
            elab = getattr(event, "Elab")
            sim_angle = elab[0]
        except Exception:
            continue

        for method in ["ransac", "gmm"]:
            try:
                reco_angles = getattr(event, f"{method}_angles")
                phi_angles = getattr(event, f"{method}_phi_angles")
                start = getattr(event, f"{method}_start")
                end = getattr(event, f"{method}_end")
                inter = getattr(event, f"{method}_inter")
            except Exception:
                continue

            # --- Determine if this event FAILS any selection ---
            # Default assume passed
            is_rejected = False

            # 1. Must have exactly one reconstructed track
            if len(reco_angles) != 1 or len(phi_angles) != 1:
                is_rejected = True
            else:
                phi = phi_angles[0]
                start_v = extract_1track_vector(start)
                end_v = extract_1track_vector(end)
                inter_v = extract_1track_vector(inter)
                angle_diff = sim_angle - reco_angles[0]

                # If any of these fail → rejected
                if 70 <= abs(phi) <= 110:
                    is_rejected = True
                if not (is_inside_volume(start_v) and is_inside_volume(end_v) and is_inside_volume(inter_v)):
                    is_rejected = True
                if not is_outside_beam_zone(end_v[1]):
                    is_rejected = True
                if not (angle_hist_range[0] < angle_diff < angle_hist_range[1]):
                    is_rejected = True

            if not is_rejected:
                continue  # skip accepted, we only want rejected

            # --- Record ARIs for rejected event ---
            for ari_branch in [b for b in required_branches if method in b]:
                vals = getattr(event, ari_branch, [])
                for v in vals:
                    ari_data[ari_branch].append(float(v))

    f.Close()
    return ari_data

# --- Main loop ---
for energy in excitation_energies:
    for cm in cm_angles:
        for suppress in suppression_factor:
            pattern = os.path.join(base_path, f"final_sim_5000_{energy}mev_{cm}cm_*_*_{suppress}.root")
            files = glob.glob(pattern)
            if not files:
                print(f"No files for {energy}MeV {cm}cm suppress {suppress}")
                continue

            combined = {b: [] for b in required_branches}
            for fpath in files:
                d = process_file_for_rejected_aris(fpath)
                for k in combined:
                    combined[k] += d.get(k, [])

            # --- Create canvas ---
            canvas = ROOT.TCanvas(f"c_rej_{energy}_{cm}", f"Rejected ARI {energy} MeV, {cm}°", 1000, 700)
            colors = [ROOT.kBlue, ROOT.kMagenta, ROOT.kRed, ROOT.kOrange+7, ROOT.kGreen+2]

            legend = ROOT.TLegend(0.6, 0.6, 0.88, 0.88)
            legend.SetBorderSize(0)
            legend.SetFillStyle(0)
            legend.SetTextSize(0.03)

            max_y = 0
            hist_list = []

            # --- Draw all histograms on one pad ---
            for i, branch in enumerate(required_branches):
                vals = combined[branch]
                hist = ROOT.TH1F(f"h_{branch}_{i}", f";ARI;Counts", n_bins, ari_xrange[0], ari_xrange[1])
                for v in vals:
                    hist.Fill(v)
                hist.SetLineColor(colors[i])
                hist.SetLineWidth(2)
                hist.SetStats(0)

                max_y = max(max_y, hist.GetMaximum())
                legend.AddEntry(hist, f"{branch} (mean={hist.GetMean():.2f}, entries={hist.GetEntries()})", "l")
                hist_list.append(hist)

            for hist in hist_list:
                hist.SetMaximum(max_y * 1.1)

            hist_list[0].Draw("HIST")
            for hist in hist_list[1:]:
                hist.Draw("HIST SAME")

            legend.Draw()
            canvas.Update()

            out_png = f"noe1e2_rejected_ARI_all_{energy}MeV_{cm}cm_{suppress}.png"
            canvas.SaveAs(out_png)
            print(f"✅ Saved {out_png}")
