#!/bin/env python
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.facecolor":  (1,1,1,0)})
from sklearn.metrics import roc_curve

from utils import GatorConfig

parser = argparse.ArgumentParser(description="Make standard plots")
parser.add_argument("config_json", type=str, help="config JSON")
parser.add_argument(
    "--epoch", type=int, default=50, metavar="N",
    help="training epoch of model to use for inference (default: 50)"
)
parser.add_argument(
    "--loss_logy", action="store_true",
    help="make y-axis of loss curve log-scale"
)
parser.add_argument(
    "--loss_N", type=int, default=1, 
    help="plot every Nth epoch (default: 1)"
)
args = parser.parse_args()

config = GatorConfig.from_json(args.config_json)
plots_dir = f"{config.base_dir}/{config.name}/plots"
os.makedirs(plots_dir, exist_ok=True)

# Get history JSON
history = {}
with open(config.get_outfile(tag="history", ext="json", short=True)) as f:
    history = json.load(f)

# Get testing inferences
test_csv = config.get_outfile(subdir="inferences", tag="test_inferences", epoch=args.epoch, ext="csv")
test_df = pd.read_csv(test_csv)

# Get training inferences
train_csv = config.get_outfile(subdir="inferences", tag="train_inferences", epoch=args.epoch, ext="csv")
train_df = pd.read_csv(train_csv)

# Merge testing and training dfs
total_df = pd.concat([test_df, train_df])

# Plot loss curve -----------------------------------------------------------------------
fig, axes = plt.subplots(figsize=(16, 12))

epochs = np.arange(1, len(history["test_loss"]) + 1)

axes.plot(epochs[::args.loss_N], history["test_loss"][::args.loss_N], label="test", color="C1");
axes.plot(epochs[::args.loss_N], history["train_loss"][::args.loss_N], label="train", color="C0");

axes.axvline(args.epoch, color="k", alpha=0.25)

axes.tick_params(axis="both", which="both", direction="in", labelsize=32, top=True, right=True)
axes.set_xlabel("Epoch", size=32);
axes.set_ylabel("Avg. Loss", size=32);
if args.loss_logy:
    axes.set_yscale("log")
axes.autoscale()
axes.legend(fontsize=24)

plt.savefig(f"{plots_dir}/loss_epoch{args.epoch}.png", bbox_inches="tight")
plt.savefig(f"{plots_dir}/loss_epoch{args.epoch}.pdf", bbox_inches="tight")
print(f"Wrote loss curve to {plots_dir}/loss_epoch{args.epoch}.png")
plt.close()
# ---------------------------------------------------------------------------------------


# Plot ROC curve ------------------------------------------------------------------------
fig, axes = plt.subplots(figsize=(12,12))

# Plot training ROC curve
fpr, tpr, thresh = roc_curve(train_df.label, train_df.score)
axes.plot(fpr, tpr, label=f"train (AUC = {np.trapz(tpr, fpr):.2f})", color="C0")

# Plot testing ROC curve
fpr, tpr, thresh = roc_curve(test_df.label, test_df.score)
axes.plot(fpr, tpr, label=f"test (AUC = {np.trapz(tpr, fpr):.2f})", color="C1")

# Format axes
axes.tick_params(axis="both", which="both", direction="in", labelsize=32, top=True, right=True)
axes.set_xlabel("Background efficiency", size=32);
axes.set_ylabel("Signal efficiency", size=32);
axes.legend(fontsize=24);

plt.savefig(f"{plots_dir}/roc_epoch{args.epoch}.png", bbox_inches="tight")
plt.savefig(f"{plots_dir}/roc_epoch{args.epoch}.pdf", bbox_inches="tight")
print(f"Wrote ROC curve to {plots_dir}/roc_epoch{args.epoch}.png")
plt.close()
# ---------------------------------------------------------------------------------------


# Plot score histogram ------------------------------------------------------------------
fig, axes = plt.subplots(figsize=(12, 12))

bins = np.linspace(0, 1, 101)

test_n_real = np.sum(test_df.label == 1)
test_n_fake = np.sum(test_df.label == 0)
axes.hist(
    test_df[test_df.label == 1].score, 
    weights=np.ones(test_n_real)/test_n_real, 
    bins=bins,
    histtype="step",
    linewidth=2,
    label="test (sig)"
);
axes.hist(
    test_df[test_df.label == 0].score, 
    weights=np.ones(test_n_fake)/test_n_fake, 
    bins=bins,
    histtype="step",
    linewidth=2,
    label="test (bkg)"
);
axes.set_yscale("log");
axes.legend(fontsize=16)

axes.tick_params(axis="both", which="both", direction="in", labelsize=20, top=True, right=True)
axes.set_xlabel("score", size=20);
axes.set_ylabel("a.u.", size=20);

plt.savefig(f"{plots_dir}/scores_epoch{args.epoch}.png", bbox_inches="tight")
plt.savefig(f"{plots_dir}/scores_epoch{args.epoch}.pdf", bbox_inches="tight")
print(f"Wrote scores histogram to {plots_dir}/scores_epoch{args.epoch}.png")
plt.close()
# ---------------------------------------------------------------------------------------

# --- Wrap up ---
print(f"\nDone. All plots can be found here: {plots_dir}\n")
