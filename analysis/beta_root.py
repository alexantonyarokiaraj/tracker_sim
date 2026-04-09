import numpy as np
import ROOT

# Sample data (replace this with your actual NumPy array)
data = np.load('/home2/user/u0100486/linux/doctorate/github/tracker_sim/output/text_files/beta_array_new.npy')
# Extract unique values from column 1 (cm angle)
unique_angles = np.unique(data[:, 0])

# Create a TMultiGraph to hold all TGraph objects
multi_graph = ROOT.TMultiGraph()

# Create a TCanvas to display the graph
canvas = ROOT.TCanvas("canvas", "Column 4 vs Column 2", 800, 600)

# Define colors and markers for different angles
colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, ROOT.kOrange]
markers = [20, 21, 22, 23, 24]

# Loop through unique angles and create TGraph for each
for i, angle in enumerate(unique_angles):
    # Filter rows where column 1 matches the current angle
    subset = data[data[:, 0] == angle]

    # Extract column 2 (x-axis) and column 4 (y-axis)
    x_values = subset[:, 1].astype(float)   # Ensure float type
    y_values = subset[:, 3].astype(float)  # Ensure float type

    # Create a TGraph
    graph = ROOT.TGraph(len(x_values), x_values, y_values)
    graph.SetTitle(f"CM Angle = {angle}")
    graph.SetMarkerColor(colors[i])
    graph.SetMarkerStyle(markers[i])
    graph.SetLineColor(colors[i])

    # Add the TGraph to the TMultiGraph
    multi_graph.Add(graph)

# Draw the TMultiGraph
multi_graph.Draw("APL")  # "A" for axis, "P" for points, "L" for lines

# Set X-axis title with LaTeX formatting
multi_graph.GetXaxis().SetTitle(r"\gamma_{kb}")  # X-axis: beta in mm

# Set Y-axis title with LaTeX formatting
multi_graph.GetYaxis().SetTitle(r"\mu(\theta_{labs} - \theta_{labr})^{\circ}")

# Set axis titles
# multi_graph.GetXaxis().SetTitle("Column 2")
# multi_graph.GetYaxis().SetTitle("Column 4")
# multi_graph.SetTitle("Column 4 vs Column 2 for Different CM Angles")

# Add a legend
legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
for i, angle in enumerate(unique_angles):
    # Use LaTeX formatting for the legend entry
    legend.AddEntry(multi_graph.GetListOfGraphs().At(i), r"\theta_{cm} = " + str(int(angle)) + r"^{\circ}", "lp")
legend.Draw()

# Update the canvas
canvas.Update()
canvas.WaitPrimitive()