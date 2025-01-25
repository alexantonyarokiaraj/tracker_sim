import ROOT
import numpy as np

# Define arrays for excitation energy and cm angle
excitation_energies = [0]  # Example values
cm_angles = [3]  # Example values
benchmark_range = 40  # The range used as a benchmark for track selection

# Loop over excitation energy and cm angle
for energy in excitation_energies:
    for cm_angle in cm_angles:
        # Construct the ROOT file names dynamically for both ranges
        file_1 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta/beta_sim_5000_{energy}mev_{cm_angle}cm_1_2500.root"
        file_2 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta/beta_sim_5000_{energy}mev_{cm_angle}cm_2501_5000.root"

        print(f"Processing files: {file_1} and {file_2}")

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
        h_gmm_end_x = ROOT.TH1F("h_gmm_end_x", "gmm_end X;gmm_end_x;Counts", 256, 0, 256)
        h_gmm_end_y = ROOT.TH1F("h_gmm_end_y", "gmm_end Y;gmm_end_y;Counts", 256, 0, 256)
        h_gmm_end_z = ROOT.TH1F("h_gmm_end_z", "gmm_end Z;gmm_end_z;Counts", 256, 0, 256)

        # Open the ROOT files
        root_file_1 = ROOT.TFile.Open(file_1, "READ")
        root_file_2 = ROOT.TFile.Open(file_2, "READ")

        if not root_file_1 or root_file_1.IsZombie() or not root_file_2 or root_file_2.IsZombie():
            print(f"Error: Cannot open one or both files {file_1} or {file_2}")
            continue

        # Get the TTree from both files
        tree_name = "events"  # Change if your tree has a different name
        tree_1 = root_file_1.Get(tree_name)
        tree_2 = root_file_2.Get(tree_name)

        if not tree_1 or not tree_2:
            print(f"Error: TTree '{tree_name}' not found in one of the files.")
            root_file_1.Close()
            root_file_2.Close()
            continue

        # Dictionary to store selected angles for each range (across all events)
        angles_by_range = {}

        counter = 0

        # Loop over both trees
        for tree in [tree_1, tree_2]:
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

                # Convert ransac_end to a NumPy array
                gmm_end = np.array(event.gmm_end)

                # Ensure ransac_end has the correct shape (N_tracks, 3)
                if gmm_end.size % 3 != 0:
                    print(f"Warning: Unexpected gmm_end size {gmm_end.size} for event {event.eventid}")
                    continue  # Skip event if the size is not a multiple of 3

                gmm_end = gmm_end.reshape(-1, 3)  # Reshape into (N_tracks, 3) format

                # Check if at least one track has (x, y, z) within [10, 246] mm
                # if not np.any((10 <= gmm_end[:, 0]) & (gmm_end[:, 0] <= 246) &
                #             (10 <= gmm_end[:, 1]) & (gmm_end[:, 1] <= 246) &
                #             (10 <= gmm_end[:, 2]) & (gmm_end[:, 2] <= 246)):
                #     continue  # Skip this event if no track meets the condition

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
                    h_gmm_end_x.Fill(event.gmm_end[0])
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



        print('angle', cm_angle)
        # Create a new ROOT canvas
        canvas = ROOT.TCanvas(f"Canvas_{energy}_{cm_angle}", f"Energy={energy} MeV, Angle={cm_angle}°", 800, 600)
        canvas.Divide(5, 2)  # Divide canvas into sub-pads (adjust if needed)

        histograms = {}  # Dictionary to store histograms
        pad_num = 1  # Track which pad to use

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

            # Draw histogram
            canvas.cd(pad_num)
            histograms[range_key].Draw()
            pad_num += 1

        canvas.Update()

        # Create a canvas named "canvas1" to draw histograms
        canvas1 = ROOT.TCanvas("canvas1", "Histograms", 1200, 800)
        canvas1.Divide(4, 4)  # 4x4 grid

        # Draw each histogram on a different pad
        canvas1.cd(1); h_verX.Draw()
        canvas1.cd(2); h_verY.Draw()
        canvas1.cd(3); h_verZ.Draw()
        canvas1.cd(4); h_gmm_phi_angles.Draw()

        canvas1.cd(5); h_gmm_inter_x.Draw()
        canvas1.cd(6); h_gmm_inter_y.Draw()
        canvas1.cd(7); h_gmm_inter_z.Draw()

        canvas1.cd(8); h_gmm_start_x.Draw()
        canvas1.cd(9); h_gmm_start_y.Draw()
        canvas1.cd(10); h_gmm_start_z.Draw()

        canvas1.cd(11); h_gmm_end_x.Draw()
        canvas1.cd(12); h_gmm_end_y.Draw()
        canvas1.cd(13); h_gmm_end_z.Draw()

        # Update the canvas and show the plots
        canvas1.Update()

        canvas.WaitPrimitive()  # Wait before moving to the next combination
        canvas1.WaitPrimitive()



        root_file_1.Close()
        root_file_2.Close()
