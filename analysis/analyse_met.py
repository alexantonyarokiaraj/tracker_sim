import ROOT
import glob
import os
import sys

# Create histograms
hist_bb = ROOT.TH1F("hist_bb", "GMM BB Metric ", 100, 0, 1)
hist_bt = ROOT.TH1F("hist_bt", "GMM BT Metric ", 100, 0, 1)
hist_tt = ROOT.TH1F("hist_tt", "GMM TT Metric ", 100, 0, 1)
hist_td = ROOT.TH1F("hist_td", "GMM TD Metric ", 50, 0, 100)

# Define excitation energy and CM values
excitation_energies = [0, 5, 10, 15, 20, 25]
cm_values = [1,2,3,4,5]

# Loop through excitation energy and CM combinations
for ex in excitation_energies:
    for cm in cm_values:
        file_path = "/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/optimize/alpha/"
        file_pattern = f"alpha_sim_5000_{ex}mev_{cm}cm_*.root"
        file_list = glob.glob(file_path + file_pattern)

        chain = ROOT.TChain("events")

        # Add all files to the TChain
        for filename in file_list:
            chain.Add(filename)

        # Loop through entries and fill histograms
        for entry in range(chain.GetEntries()):
            chain.GetEntry(entry)
            if chain.eventid[0] > 0:
                # Fill histograms while ensuring matching vector sizes
                if len(chain.gmm_bb_metric) == len(chain.gmm_bb_size1) == len(chain.gmm_bb_size2):
                    for i in range(len(chain.gmm_bb_metric)):
                        if chain.gmm_bb_size1[i] > 40 and chain.gmm_bb_size2[i] > 40:
                            hist_bb.Fill(chain.gmm_bb_metric[i])

                if len(chain.gmm_bt_metric) == len(chain.gmm_bt_size1) == len(chain.gmm_bt_size2):
                    for i in range(len(chain.gmm_bt_metric)):
                        if chain.gmm_bt_size1[i] > 10 and chain.gmm_bt_size2[i] > 10:
                            hist_bt.Fill(chain.gmm_bt_metric[i])

                if len(chain.gmm_tt_metric) == len(chain.gmm_tt_size1) == len(chain.gmm_tt_size2):
                    for i in range(len(chain.gmm_tt_metric)):
                        if chain.gmm_tt_size1[i] > 20 and chain.gmm_tt_size2[i] > 20:
                            hist_tt.Fill(chain.gmm_tt_metric[i])

                if len(chain.gmm_td_metric) == len(chain.gmm_td_size1) == len(chain.gmm_td_size2):
                    for i in range(len(chain.gmm_td_metric)):
                        if chain.gmm_td_size1[i] > 10 and chain.gmm_td_size2[i] > 10:
                            hist_td.Fill(chain.gmm_td_metric[i])

# Create canvas and draw histograms
canvas = ROOT.TCanvas("canvas", "Histograms", 800, 800)
canvas.Divide(2, 2)

canvas.cd(1)
hist_bb.SetLineColor(ROOT.kRed)
hist_bb.Draw()

canvas.cd(2)
hist_bt.SetLineColor(ROOT.kBlue)
hist_bt.Draw()

canvas.cd(3)
hist_tt.SetLineColor(ROOT.kGreen)
hist_tt.Draw()

canvas.cd(4)
hist_td.SetLineColor(ROOT.kMagenta)
hist_td.Draw()

canvas.Update()
canvas.WaitPrimitive()
