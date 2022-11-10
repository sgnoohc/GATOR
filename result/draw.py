#!/bin/env python

import plottery_wrapper as p

p.dump_plot(
        fnames = [
            "bkg.root",
            ],
        sig_fnames = [
            "sig.root",
            ],
        legend_labels = [
            "fake LS",
            ],
        signal_labels = [
            "true LS",
            ],
        extraoptions={
            "yaxis_log":True,
            # "signal_scale":"auto",
            "lumi_value":-1,
            "yaxis_label":"# of LS",
            "xaxis_label":"GNN output scores",
            },
        )

