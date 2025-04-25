# Code to analyse beta

import ROOT
import numpy as np

# Define arrays for excitation energy and cm angle
excitation_energies = [10]  # Example values
cm_angles = [1,2,3,4,5]  # Example values
benchmark_range = 40  # The range used as a benchmark for track selection

# Loop over excitation energy and cm angle
for energy in excitation_energies:
    beta_numpy_array = []
    for cm_angle in cm_angles:
        # Construct the ROOT file names dynamically for both ranges
        file_1 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_1_312.root"
        file_2 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_313_624.root"
        file_3 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_625_936.root"
        file_4 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_937_1248.root"
        file_5 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_1249_1560.root"
        file_6 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_1561_1872.root"
        file_7 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_1873_2184.root"
        file_8 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_2185_2496.root"
        file_9 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_2497_2808.root"
        file_10 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_2809_3120.root"
        file_11 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_3121_3432.root"
        file_12 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_3433_3744.root"
        file_13 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_3745_4056.root"
        file_14 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_4057_4368.root"
        file_15 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_4369_4680.root"
        file_16 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta_single_new/beta_sim_5000_{energy}mev_{cm_angle}cm_4681_5000.root"

        # Define histograms
        h_verX = ROOT.TH1F("h_verX", "verX Distribution;verX;Counts", 256, 0, 256)
        h_verY = ROOT.TH1F("h_verY", "verY Distribution;verY;Counts", 256, 0, 256)
        h_verZ = ROOT.TH1F("h_verZ", "verZ Distribution;verZ;Counts", 256, 0, 256)

        h_gmm_phi_angles = ROOT.TH1F("h_gmm_phi_angles", "gmm_phi_angles Distribution;gmm_phi_angles;Counts", 360, -180, 180)

        # Histograms for gmm_inter (x, y, z)
        h_gmm_inter_x = ROOT.TH1F("h_gmm_inter_x", "gmm_inter X;gmm_inter_x;Counts", 256, 0, 256)
        h_gmm_inter_y = ROOT.TH1F("h_gmm_inter_y", "gmm_inter Y;gmm_inter_y;Counts", 256, 0, 256)
        h_gmm_inter_z = ROOT.TH1F("h_gmm_inter_z", "gmm_inter Z;gmm_inter_z;Counts", 256, 0, 256)

        # Histograms for gmm_start (x, y, z)
        h_gmm_start_x = ROOT.TH1F("h_gmm_start_x", "gmm_start X;gmm_start_x;Counts", 256, 0, 256)
        h_gmm_start_y = ROOT.TH1F("h_gmm_start_y", "gmm_start Y;gmm_start_y;Counts", 256, 0, 256)
        h_gmm_start_z = ROOT.TH1F("h_gmm_start_z", "gmm_start Z;gmm_start_z;Counts", 256, 0, 256)

        # Histograms for gmm_end (x, y, z)
        h_gmm_end_x = ROOT.TH1F("h_gmm_end_x", "gmm_end X;gmm_end_x;Counts", 256, -256, 256)
        h_gmm_end_y = ROOT.TH1F("h_gmm_end_y", "gmm_end Y;gmm_end_y;Counts", 256, 0, 256)
        h_gmm_end_z = ROOT.TH1F("h_gmm_end_z", "gmm_end Z;gmm_end_z;Counts", 256, 0, 256)

        # Open the ROOT files
        root_file_1 = ROOT.TFile.Open(file_1, "READ")
        root_file_2 = ROOT.TFile.Open(file_2, "READ")
        root_file_3 = ROOT.TFile.Open(file_3, "READ")
        root_file_4 = ROOT.TFile.Open(file_4, "READ")
        root_file_5 = ROOT.TFile.Open(file_5, "READ")
        root_file_6 = ROOT.TFile.Open(file_6, "READ")
        root_file_7 = ROOT.TFile.Open(file_7, "READ")
        root_file_8 = ROOT.TFile.Open(file_8, "READ")
        root_file_9 = ROOT.TFile.Open(file_9, "READ")
        root_file_10 = ROOT.TFile.Open(file_10, "READ")
        root_file_11 = ROOT.TFile.Open(file_11, "READ")
        root_file_12 = ROOT.TFile.Open(file_12, "READ")
        root_file_13 = ROOT.TFile.Open(file_13, "READ")
        root_file_14 = ROOT.TFile.Open(file_14, "READ")
        root_file_15 = ROOT.TFile.Open(file_15, "READ")
        root_file_16 = ROOT.TFile.Open(file_16, "READ")

        # Get the TTree from both files
        tree_name = "events"  # Change if your tree has a different name
        tree_1 = root_file_1.Get(tree_name)
        tree_2 = root_file_2.Get(tree_name)
        tree_3 = root_file_3.Get(tree_name)
        tree_4 = root_file_4.Get(tree_name)
        tree_5 = root_file_5.Get(tree_name)
        tree_6 = root_file_6.Get(tree_name)
        tree_7 = root_file_7.Get(tree_name)
        tree_8 = root_file_8.Get(tree_name)
        tree_9 = root_file_9.Get(tree_name)
        tree_10 = root_file_10.Get(tree_name)
        tree_11 = root_file_11.Get(tree_name)
        tree_12 = root_file_12.Get(tree_name)
        tree_13 = root_file_13.Get(tree_name)
        tree_14 = root_file_14.Get(tree_name)
        tree_15 = root_file_15.Get(tree_name)
        tree_16 = root_file_16.Get(tree_name)

        if not tree_1 or not tree_2 or not tree_3 or not tree_4 or not tree_5 or not tree_6 or not tree_7 or not tree_8 or not tree_9 or not tree_10 or not tree_11 or not tree_12 or not tree_13 or not tree_14 or not tree_15 or not tree_16:
            print(f"Error: TTree '{tree_name}' not found in one of the files.")
            root_file_1.Close()
            root_file_2.Close()
            root_file_3.Close()
            root_file_4.Close()
            root_file_5.Close()
            root_file_6.Close()
            root_file_7.Close()
            root_file_8.Close()
            root_file_9.Close()
            root_file_10.Close()
            root_file_11.Close()
            root_file_12.Close()
            root_file_13.Close()
            root_file_14.Close()
            root_file_15.Close()
            root_file_16.Close()
            continue

        # Dictionary to store selected angles for each range (across all events)
        angles_by_range = {}

        counter = 0

        # Loop over both trees
        for tree in [tree_1, tree_2, tree_3, tree_4, tree_5, tree_6, tree_7, tree_8, tree_9, tree_10, tree_11, tree_12, tree_13, tree_14, tree_15, tree_16]:
            for event in tree:
                # print('Event->', event.eventid)
                if counter > 5200:
                    break
                counter += 1
                beta_gmm_tracks = list(event.beta_gmm_tracks)  # Track IDs (outer dictionary keys)
                beta_gmm_counts = list(event.beta_gmm_counts)  # Number of (range, angle) pairs per track
                beta_gmm = list(event.beta_gmm)  # Inner dictionary keys (ranges)
                beta_gmm_angle = list(event.beta_gmm_angle)  # Inner dictionary values (angles)

                phi_angles = np.array(event.gmm_phi_angles)

                # Check if any phi_angle is in the exclusion range
                if np.any((70 <= phi_angles) & (phi_angles <= 110)) or np.any((-110 <= phi_angles) & (phi_angles <= -70)):
                    continue  # Skip this event if any phi_angle is within the exclusion range

                gmm_end = np.array(event.gmm_end)

                if gmm_end.size % 3 != 0:
                    print(f"Warning: Unexpected gmm_end size {gmm_end.size} for event {event.eventid}")
                    continue  # Skip event if the size is not a multiple of 3

                gmm_end = gmm_end.reshape(-1, 3)  # Reshape into (N_tracks, 3) format

                # Check if at least one track has (x, y, z) within [10, 246] mm
                if not np.any((10 <= gmm_end[:, 0]) & (gmm_end[:, 0] <= 246) &
                            (10 <= gmm_end[:, 1]) & (gmm_end[:, 1] <= 246) &
                            (10 <= gmm_end[:, 2]) & (gmm_end[:, 2] <= 246)):
                    continue  # Skip this event if no track meets the condition

                Elab_value = event.Elab[0]  # Get Elab for the current event

                for x in event.verX:
                    h_verX.Fill(x)
                for y in event.verY:
                    h_verY.Fill(y)
                for z in event.verZ:
                    h_verZ.Fill(z)
                for phi in event.gmm_phi_angles:
                    h_gmm_phi_angles.Fill(phi)

                if len(event.gmm_inter) == 3:
                    h_gmm_inter_x.Fill(event.gmm_inter[0])
                    h_gmm_inter_y.Fill(event.gmm_inter[1])
                    h_gmm_inter_z.Fill(event.gmm_inter[2])


                if len(event.gmm_start) == 3:
                    h_gmm_start_x.Fill(event.gmm_start[0])
                    h_gmm_start_y.Fill(event.gmm_start[1])
                    h_gmm_start_z.Fill(event.gmm_start[2])


                if len(event.gmm_end) == 3:
                    h_gmm_end_x.Fill(event.gmm_end[0]-event.gmm_start[0])
                    h_gmm_end_y.Fill(event.gmm_end[1])
                    h_gmm_end_z.Fill(event.gmm_end[2])


                # Reconstruct the 2D dictionary for this event
                index = 0
                gmm_dict = {}

                for i, track_id in enumerate(beta_gmm_tracks):
                    num_entries = int(beta_gmm_counts[i])  # Convert to integer
                    sub_dict = dict(zip(beta_gmm[index:index + num_entries],
                                        beta_gmm_angle[index:index + num_entries]))
                    gmm_dict[track_id] = sub_dict
                    index += num_entries
                    # value = sub_dict.get(40.0, 0)
                    # if int(value) - int(Elab_value) < -5 and int(value) - int(Elab_value) > -10:
                    #     print(event.eventid, int(value), int(Elab_value))

                 # Step 1: Find the track whose angle at range 40 is closest to Elab
                closest_track = None
                min_difference = float("inf")

                for track_id, ranges in gmm_dict.items():
                    # print(track_id, ":" ,ranges)
                    if benchmark_range in ranges:
                        angle_at_40 = ranges[benchmark_range]
                        # print('Angle at 40', angle_at_40)
                        difference = abs(angle_at_40 - Elab_value)
                        if difference < min_difference:
                            min_difference = difference
                            closest_track = track_id


                if closest_track is None:
                    continue  # No valid track found, skip this event

                # Step 2: Use only the closest track's (range, angle) pairs
                selected_ranges = gmm_dict[closest_track]
                # print(selected_ranges)

                 # Store the angles for each range
                for range_key, angle in selected_ranges.items():
                    if range_key not in angles_by_range:
                        angles_by_range[range_key] = []
                    angles_by_range[range_key].append(Elab_value-angle)

        histograms = {}  # Dictionary to store histograms

        # Create histograms for each range
        for range_key, angles in angles_by_range.items():
            angles_np = np.array(angles)  # Convert to numpy array

            # Define histogram properties
            hist_name = f"hist_{energy}_{cm_angle}_{range_key}"
            histograms[range_key] = ROOT.TH1F(hist_name, f"Range {range_key} (Energy={energy} MeV, Angle={cm_angle}°);Angle;Counts",
                                              80, -20, 20)  # 50 bins

            # Fill histogram with data
            for angle in angles_np:
                histograms[range_key].Fill(angle)

            beta_numpy_array.append([cm_angle, range_key, histograms[range_key].GetMean(), histograms[range_key].GetStdDev()])


        root_file_1.Close()
        root_file_2.Close()
        root_file_3.Close()
        root_file_4.Close()
        root_file_5.Close()
        root_file_6.Close()
        root_file_7.Close()
        root_file_8.Close()
        root_file_9.Close()
        root_file_10.Close()
        root_file_11.Close()
        root_file_12.Close()
        root_file_13.Close()
        root_file_14.Close()
        root_file_15.Close()
        root_file_16.Close()
        # canvas = ROOT.TCanvas("canvas", "2D Histogram", 800, 600)
        # h_gmm_end_x.Draw()  # "COLZ" for color mapping (heatmap)
        # canvas.Update()
        # canvas.Draw()
        # canvas.WaitPrimitive()
    np.save('/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/text_files/beta_array_new.npy', np.array(beta_numpy_array))
