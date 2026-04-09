import ROOT
import numpy as np

# Define arrays for excitation energy and cm angle
excitation_energies = [10]  # Example values
cm_angles = [1,2,3,4,5]  # Example values

for energy in excitation_energies:
    responsibility_array = []
    for cm_angle in cm_angles:
        print(f"\nProcessing cm_angle = {cm_angle}...\n")
        histograms = {}
        file_paths = [
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_1_312.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_313_624.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_625_936.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_937_1248.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_1249_1560.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_1561_1872.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_1873_2184.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_2185_2496.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_2497_2808.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_2809_3120.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_3121_3432.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_3433_3744.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_3745_4056.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_4057_4368.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_4369_4680.root",
            f"/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/optimize/gamma_single_new/gamma_sim_5000_{energy}mev_{cm_angle}cm_4681_5000.root",
        ]

        histograms = {}  # Store histograms per unique responsibility
        responsibility_dict = {}


        for file_path in file_paths:
            root_file = ROOT.TFile.Open(file_path, "READ")

            if not root_file or root_file.IsZombie():
                print(f"Error: Could not open {file_path}")
                continue

            tree = root_file.Get("events")
            if not tree:
                print(f"Error: TTree 'events' not found in {file_path}")
                root_file.Close()
                continue

            for event in tree:
                if event.eventid[0] >= 0:
                    # print(event.eventid)
                    # **1️⃣ Exclusion based on phi_angles**

                    phi_angles = np.array(event.gmm_phi_angles)
                    if np.any((70 <= phi_angles) & (phi_angles <= 110)) or np.any((-110 <= phi_angles) & (phi_angles <= -70)):
                        continue  # Skip event

                    # **2️⃣ Extract and reshape gmm_end (x, y, z)**
                    gmm_end = np.array(event.gmm_end)
                    if gmm_end.size % 3 != 0:
                        print(f"Warning: Unexpected gmm_end size {gmm_end.size} for event {event.eventid}")
                        continue

                    gmm_end = gmm_end.reshape(-1, 3)  # Reshape into (N_tracks, 3)

                    # **3️⃣ Track selection: (x, y, z) must be in range [10, 246] mm**
                    if not np.any((10 <= gmm_end[:, 0]) & (gmm_end[:, 0] <= 246) &
                                (10 <= gmm_end[:, 1]) & (gmm_end[:, 1] <= 246) &
                                (10 <= gmm_end[:, 2]) & (gmm_end[:, 2] <= 246)):
                        continue  # Skip event

                    # Extract Elab (single-element vector)
                    Elab = float(event.Elab[0]) if len(event.Elab) > 0 else 0

                    lab_angles_initial = np.array(event.gmm_angles)
                    # print(lab_angles_initial)

                    # **4️⃣ Extract responsibilities and angles**
                    gmm_resp_initial = list(event.gmm_resp)
                    gmm_resp_angle_initial = list(event.gmm_resp_angle)

                    # Step 1️⃣: Find the index of lab_angles_initial closest to Elab
                    closest_index = np.abs(lab_angles_initial - Elab).argmin()

                    # Step 2️⃣: Determine the start and end indices for extraction
                    start_idx = closest_index * 304
                    end_idx = start_idx + 304

                    gmm_resp = gmm_resp_initial[start_idx:end_idx]
                    gmm_resp_angle = gmm_resp_angle_initial[start_idx:end_idx]

                    if len(gmm_resp) != len(gmm_resp_angle):
                        print(f"Warning: Mismatched sizes in event (resp={len(gmm_resp)}, angle={len(gmm_resp_angle)})")
                        continue

                    if len(gmm_resp) > 304:
                        print("Length exceeded", len(gmm_resp),event.eventid)
                        continue

                    # Initialize a counter outside the loop to keep track of histograms
                    hist_counter = 0
                    # Fill histograms for valid events
                    for resp, angle in zip(gmm_resp, gmm_resp_angle):
                        delta = Elab - angle  # Compute Elab - angle

                        # Increment the counter for each unique histogram
                        hist_counter += 1
                        hist_key = f"hist_{hist_counter}"  # Unique key for each histogram

                        if hist_key not in histograms:

                            hist_name = f"hist_resp_{hist_counter}"  # Generate histogram name based on counter
                            histograms[hist_key] = ROOT.TH1F(hist_name, f"Elab - Angle for resp",
                                                            80, -20, 20)  # Adjust range if needed
                            histograms[hist_key].SetDirectory(0)
                            # print(f"Histogram {hist_key} created successfully.")
                            responsibility_dict[hist_key] = resp

                        if delta >= -20 and delta <= 20:
                            histograms[hist_key].Fill(delta)

            root_file.Close()  # ✅ Close ROOT file after processing
        # **6️⃣ Compute and print statistics**
        for resp, hist in histograms.items():
            mean = hist.GetMean()
            stddev = hist.GetStdDev()
            print(f"Responsibility: {responsibility_dict[resp]} -> Mean: {mean:.3f}, StdDev: {stddev:.3f}, Entries: {hist.GetEntries()}")
            responsibility_array.append([cm_angle, responsibility_dict[resp], mean, stddev])
            histograms[resp].Delete()
        histograms.clear()
    print(responsibility_array)
    np.save('/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/text_files/gamma_array_new.npy', np.array(responsibility_array))



