import ROOT
import glob
import os
import sys

# Create histograms
hist_bb = ROOT.TH1F("hist_bb", "GMM BB Metric", 100, 0, 1)
hist_bt = ROOT.TH1F("hist_bt", "GMM BT Metric", 100, 0, 1)
hist_tt = ROOT.TH1F("hist_tt", "GMM TT Metric", 100, 0, 1)
# hist_td is not needed

# Define excitation energy and CM values
excitation_energies = [10]
cm_values = [1, 2, 3, 4, 5]

# Loop through excitation energy and CM combinations
for ex in excitation_energies:
    for cm in cm_values:
        file_path = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/root_files_1/"
        file_pattern = f"final_sim_5000_{ex}mev_{cm}cm_*.root"
        file_list = glob.glob(file_path + file_pattern)
        print('file_pattern', file_pattern)
        chain = ROOT.TChain("events")

        # Add all files to the TChain
        for filename in file_list:
            chain.Add(filename)

        # Loop through entries and fill histograms
        for entry in range(chain.GetEntries()):
            chain.GetEntry(entry)
            if chain.eventid[0] > 0:
                # Fill hist_bb histogram
                if len(chain.gmm_bb_metric) == len(chain.gmm_bb_size1) == len(chain.gmm_bb_size2):
                    for i in range(len(chain.gmm_bb_metric)):
                        if chain.gmm_bb_size1[i] > 40 or chain.gmm_bb_size2[i] > 40:
                            hist_bb.Fill(chain.gmm_bb_metric[i])

                # Fill hist_bt histogram
                if len(chain.gmm_bt_metric) == len(chain.gmm_bt_size1) == len(chain.gmm_bt_size2):
                    for i in range(len(chain.gmm_bt_metric)):
                        if chain.gmm_bt_size1[i] > 10 or chain.gmm_bt_size2[i] > 10:
                            hist_bt.Fill(chain.gmm_bt_metric[i])

                # Fill hist_tt histogram
                if len(chain.gmm_tt_metric) == len(chain.gmm_tt_size1) == len(chain.gmm_tt_size2):
                    for i in range(len(chain.gmm_tt_metric)):
                        if chain.gmm_tt_size1[i] > 10 or chain.gmm_tt_size2[i] > 10:
                            hist_tt.Fill(chain.gmm_tt_metric[i])
                            if chain.gmm_tt_metric[i] < 0.1:
                                print(chain.eventid[0], ex,cm)

# Assuming hist_bb, hist_bt, and hist_tt are already defined as TH1 objects
if hist_bb.Integral() > 0:
    hist_bb.Scale(1.0 / hist_bb.Integral())

if hist_bt.Integral() > 0:
    hist_bt.Scale(1.0 / hist_bt.Integral())

if hist_tt.Integral() > 0:
    hist_tt.Scale(1.0 / hist_tt.Integral())

# Create a canvas and plot all three histograms on the same pad
canvas = ROOT.TCanvas("canvas", "Normalized Histograms", 800, 600)

# Configure and draw the histograms
hist_bb.SetLineColor(ROOT.kRed)
hist_bb.Draw("HIST")

hist_bt.SetLineColor(ROOT.kBlue)
hist_bt.Draw("HIST SAME")

hist_tt.SetLineColor(ROOT.kGreen+2)
hist_tt.Draw("HIST SAME")

# Create a legend
legend = ROOT.TLegend(0.65, 0.70, 0.88, 0.88)
legend.AddEntry(hist_bb, "#it{p}_{#it{kl}}^{(BB)}", "l")
legend.AddEntry(hist_bt, "#it{p}_{#it{kl}}^{(BE)}", "l")
legend.AddEntry(hist_tt, "#it{p}_{#it{kl}}^{(EE)}", "l")
legend.Draw()

canvas.Update()
canvas.WaitPrimitive()