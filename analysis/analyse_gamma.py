import ROOT
import numpy as np

# Define arrays for excitation energy and cm angle
excitation_energies = [10]  # Example values
cm_angles = [1, 2, 3, 4, 5]  # Example values
benchmark_range = 40  # The range used as a benchmark for track selection

# Loop over excitation energy and cm angle
for energy in excitation_energies:
    hist = {}
    canvas = ROOT.TCanvas("canvas", "2D Histogram", 1200, 300)  # Wider canvas
    canvas.Divide(len(cm_angles), 1)  # Divide into 5 columns

    for cm_angle in cm_angles:
        # ✅ Unique histogram name for each cm_angle
        hist[cm_angle] = ROOT.TH2F(f"hist_2d_{cm_angle}",
                                   f"gmm_min_res vs gmm_min_angle for {cm_angle} cm",
                                   100, 0, 1,   # X-axis: gmm_min_res
                                   360, -180, 180) # Y-axis: gmm_min_angle

        file_paths = [
            f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/gamma/gamma_sim_5000_{energy}mev_{cm_angle}cm_1_1600.root",
            f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/gamma/gamma_sim_5000_{energy}mev_{cm_angle}cm_1601_3200.root",
            f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/gamma/gamma_sim_5000_{energy}mev_{cm_angle}cm_3201_5000.root"
        ]

        for file_path in file_paths:
            root_file = ROOT.TFile.Open(file_path, "READ")

            # ✅ Check for valid ROOT file
            if not root_file or root_file.IsZombie():
                print(f"Error: Could not open {file_path}")
                continue

            tree = root_file.Get("events")  # Change if needed
            if not tree:
                print(f"Error: TTree 'events' not found in {file_path}")
                root_file.Close()
                continue

            counter = 0

            # Loop over events
            for event in tree:
                if counter > 5200:
                    break
                counter += 1

                phi_angles = np.array(event.gmm_phi_angles)

                # Exclusion range for phi angles
                if np.any((70 <= phi_angles) & (phi_angles <= 110)) or np.any((-110 <= phi_angles) & (phi_angles <= -70)):
                    continue

                # Convert gmm_end to a NumPy array
                gmm_end = np.array(event.gmm_end)

                if gmm_end.size % 3 != 0:
                    print(f"Warning: Unexpected gmm_end size {gmm_end.size} for event {event.eventid}")
                    continue

                gmm_end = gmm_end.reshape(-1, 3)  # Reshape into (N_tracks, 3)

                # Track selection based on x, y, z within [10, 246] mm
                if not np.any((10 <= gmm_end[:, 0]) & (gmm_end[:, 0] <= 246) &
                              (10 <= gmm_end[:, 1]) & (gmm_end[:, 1] <= 246) &
                              (10 <= gmm_end[:, 2]) & (gmm_end[:, 2] <= 246)):
                    continue

                gmm_min_res = list(event.gmm_min_res)  # X-axis values
                gmm_min_angle = list(event.gmm_phi_angles)  # Y-axis values

                if len(gmm_min_res) != len(gmm_min_angle):
                    print(f"Warning: Mismatched sizes (res={len(gmm_min_res)}, angle={len(gmm_min_angle)}) in event")
                    continue

                # Fill histogram
                for res, angle in zip(gmm_min_res, gmm_min_angle):
                    hist[cm_angle].Fill(res, angle)

            root_file.Close()  # ✅ Close ROOT file after processing

        # Draw histograms on canvas
        canvas.cd(cm_angle)  # Move to correct pad
        hist[cm_angle].Draw("COLZ")
        canvas.Update()

    canvas.Draw()
    input("Press Enter to continue...")  # ✅ Avoid `WaitPrimitive()`
