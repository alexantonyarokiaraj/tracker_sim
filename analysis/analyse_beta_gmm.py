import ROOT
import numpy as np

# Define arrays for excitation energy and cm angle
excitation_energies = [0,5,10,15,20,25]  # Example values
cm_angles = [1,2,3,4]  # Example values
benchmark_range = 40  # The range used as a benchmark for track selection

canvas1 = ROOT.TCanvas("c1", "Mean Angle vs. Range", 800, 600)
canvas1.Divide(2, 2)  # 2x2 grid for cm angles
mg_list = []
line_list = []

# Loop over excitation energy and cm angle
for cm_angle in cm_angles:
    legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)  # Legend for each pad
    mg = ROOT.TMultiGraph()  # MultiGraph to hold all graphs for this angle
    mg_list.append(mg)
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kOrange, ROOT.kOrange+2]
    exc_count = 0
    mean_elabs_arr = []
    for energy in excitation_energies:
        # Construct the ROOT file names dynamically for both ranges
        file_1 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta/beta_sim_5000_{energy}mev_{cm_angle}cm_1_2500.root"
        file_2 = f"/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/beta/beta_sim_5000_{energy}mev_{cm_angle}cm_2501_5000.root"

        print(f"Processing files: {file_1} and {file_2}")

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
        mean_elab = []

        # Loop over both trees
        for tree in [tree_1, tree_2]:
            for event in tree:
                # print('Event->', event.eventid)
                if counter > 5200:
                    break
                counter += 1
                beta_ransac_tracks = list(event.beta_gmm_tracks)  # Track IDs (outer dictionary keys)
                beta_ransac_counts = list(event.beta_gmm_counts)  # Number of (range, angle) pairs per track
                beta_ransac = list(event.beta_gmm)  # Inner dictionary keys (ranges)
                beta_ransac_angle = list(event.beta_gmm_angle)  # Inner dictionary values (angles)

                phi_angles = np.array(event.gmm_phi_angles)

                # Check if any phi_angle is in the exclusion range
                if np.any((70 <= phi_angles) & (phi_angles <= 110)) or np.any((-110 <= phi_angles) & (phi_angles <= -70)):
                    continue  # Skip this event if any phi_angle is within the exclusion range

                # Convert ransac_end to a NumPy array
                ransac_end = np.array(event.gmm_end)

                # Ensure ransac_end has the correct shape (N_tracks, 3)
                if ransac_end.size % 3 != 0:
                    print(f"Warning: Unexpected ransac_end size {ransac_end.size} for event {event.eventid}")
                    continue  # Skip event if the size is not a multiple of 3

                ransac_end = ransac_end.reshape(-1, 3)  # Reshape into (N_tracks, 3) format

                # Check if at least one track has (x, y, z) within [10, 246] mm
                if not np.any((10 <= ransac_end[:, 0]) & (ransac_end[:, 0] <= 246) &
                            (10 <= ransac_end[:, 1]) & (ransac_end[:, 1] <= 246) &
                            (10 <= ransac_end[:, 2]) & (ransac_end[:, 2] <= 246)):
                    continue  # Skip this event if no track meets the condition

                Elab_value = event.Elab[0]  # Get Elab for the current event
                mean_elab.append(Elab_value)

                # Reconstruct the 2D dictionary for this event
                index = 0
                ransac_dict = {}

                for i, track_id in enumerate(beta_ransac_tracks):
                    num_entries = int(beta_ransac_counts[i])  # Convert to integer
                    sub_dict = dict(zip(beta_ransac[index:index + num_entries],
                                        beta_ransac_angle[index:index + num_entries]))
                    ransac_dict[track_id] = sub_dict
                    index += num_entries

                # print(ransac_dict)
                # print(type(Elab_value))

                 # Step 1: Find the track whose angle at range 40 is closest to Elab
                closest_track = None
                min_difference = float("inf")

                for track_id, ranges in ransac_dict.items():
                    # print(track_id, ":" ,ranges)
                    if benchmark_range in ranges:
                        angle_at_40 = ranges[benchmark_range]
                        # print('Angle at 40', angle_at_40)
                        difference = abs(angle_at_40 - Elab_value)
                        if difference < min_difference:
                            min_difference = difference
                            closest_track = track_id
                # print('closest_track', closest_track)

                if closest_track is None:
                    continue  # No valid track found, skip this event

                # Step 2: Use only the closest track's (range, angle) pairs
                selected_ranges = ransac_dict[closest_track]
                # print(selected_ranges)

                 # Store the angles for each range
                for range_key, angle in selected_ranges.items():
                    if range_key not in angles_by_range:
                        angles_by_range[range_key] = []
                    angles_by_range[range_key].append(Elab_value-angle)

        print('angle', cm_angle)
        # Create a new ROOT canvas
        # canvas = ROOT.TCanvas(f"Canvas_{energy}_{cm_angle}", f"Energy={energy} MeV, Angle={cm_angle}°", 800, 600)
        # canvas.Divide(5, 2)  # Divide canvas into sub-pads (adjust if needed)

        histograms = {}  # Dictionary to store histograms
        pad_num = 1  # Track which pad to use

        # Create histograms for each range
        for range_key, angles in angles_by_range.items():
            angles_np = np.array(angles)  # Convert to numpy array

            # Define histogram properties
            hist_name = f"hist_{energy}_{cm_angle}_{range_key}"
            histograms[range_key] = ROOT.TH1F(hist_name, f"Range {range_key} (Energy={energy} MeV, Angle={cm_angle}°);Angle;Counts",
                                              40, -20, 20)  # 50 bins

            # Fill histogram with data
            for angle in angles_np:
                histograms[range_key].Fill(angle)

            # Draw histogram
            # canvas.cd(pad_num)
            # histograms[range_key].Draw()
            pad_num += 1

        # canvas.Update()

        # Create a TGraphErrors for mean vs. range
        graph = ROOT.TGraphErrors(len(histograms))
        graph.SetTitle(f"Mean Angle vs. Range (Energy={energy} MeV, Angle={angle}°)")
        graph.GetXaxis().SetTitle("Range")
        graph.GetYaxis().SetTitle("Mean Angle (degrees)")
        graph.SetMarkerColor(colors[exc_count])
        graph.SetLineColor(colors[exc_count])
        exc_count += 1

        # Loop over histogram dictionary and extract statistics
        for i, (range_key, hist) in enumerate(sorted(histograms.items())):
            mean_angle = hist.GetMean()  # Get mean from ROOT histogram
            variance_angle = hist.GetRMS() ** 2  # Get RMS (std dev) and square it to get variance
            stddev_angle = hist.GetRMS()  # Standard deviation as error bars

            # Set data points in TGraphErrors
            graph.SetPoint(i, range_key, mean_angle+np.mean(np.array(mean_elab)))
            graph.SetPointError(i, 0, stddev_angle)  # Error bars represent standard deviation


        mg_list[int(cm_angle)-1].Add(graph)
        legend.AddEntry(graph, f"Energy {energy} MeV", "P")

        mean_elabs_arr.append(np.mean(np.array(mean_elab)))


        # canvas.WaitPrimitive()  # Wait before moving to the next combination




        root_file_1.Close()
        root_file_2.Close()
    line_list.append(mean_elabs_arr)
    for index, lines in enumerate(mean_elabs_arr):
        # Create a TGraph to represent each horizontal line
        x_values = np.array([mg_list[int(cm_angle)-1].GetXaxis().GetXmin(), mg_list[int(cm_angle)-1].GetXaxis().GetXmax()])
        y_values = np.array([lines, lines])  # Same y value for both points to make a horizontal line

        # Create TGraph for the horizontal line
        horizontal_line_graph = ROOT.TGraph(2, x_values, y_values)

        # Set the properties for the horizontal line graph
        horizontal_line_graph.SetLineColor(colors[index])  # Red color for the horizontal line
        horizontal_line_graph.SetLineStyle(2)  # Dashed line style
        horizontal_line_graph.SetLineWidth(2)  # Set the line width

        # Add the horizontal line graph to the TMultiGraph
        mg_list[int(cm_angle)-1].Add(horizontal_line_graph)


canvas1.cd(1)
ROOT.gPad.SetGrid()




mg_list[0].Draw("ALP")
mg_list[0].GetXaxis().SetTitle("Range")
mg_list[0].GetYaxis().SetTitle("Mean Angle (degrees)")

canvas1.cd(2)
ROOT.gPad.SetGrid()
mg_list[1].Draw("ALP")
mg_list[1].GetXaxis().SetTitle("Range")
mg_list[1].GetYaxis().SetTitle("Mean Angle (degrees)")


canvas1.cd(3)
ROOT.gPad.SetGrid()
mg_list[2].Draw("ALP")
mg_list[2].GetXaxis().SetTitle("Range")
mg_list[2].GetYaxis().SetTitle("Mean Angle (degrees)")


canvas1.cd(4)
ROOT.gPad.SetGrid()
mg_list[3].Draw("ALP")
mg_list[3].GetXaxis().SetTitle("Range")
mg_list[3].GetYaxis().SetTitle("Mean Angle (degrees)")


canvas1.Update()
canvas1.WaitPrimitive()  # Wait for user to close before moving to next combination