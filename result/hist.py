#!/bin/env python

import ROOT as r

f_sig = r.TFile("sig.root", "recreate")
f_bkg = r.TFile("bkg.root", "recreate")

for epoch in range(1, 11):

    f = open("csvs/data_epoch{}.csv".format(epoch))
    hname = "h_epoch{}".format(epoch)

    h_sig = r.TH1F(hname, hname, 180, 0., 1.)
    h_bkg = r.TH1F(hname, hname, 180, 0., 1.)

    lines = f.readlines()

    for line in lines:

        truth = int(float(line.split(",")[0]))
        output = float(line.split(",")[1])

        if truth:
            h_sig.Fill(output)
        else:
            h_bkg.Fill(output)

    f_sig.cd()
    h_sig.SetDirectory(f_sig)
    h_sig.Write()

    f_bkg.cd()
    h_bkg.SetDirectory(f_bkg)
    h_bkg.Write()

