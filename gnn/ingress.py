#!/bin/env python

import os
import argparse

import uproot
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from utils import GatorConfig, SimpleProgress

class Plots:
    def __init__(self, config):
        self.true_hists = {}
        self.true_hists.update({f: None for f in config.ingress.edge_features})
        self.true_hists.update({f"{f}_idx0": None for f in config.ingress.node_features})
        self.true_hists.update({f"{f}_idx1": None for f in config.ingress.node_features})
        self.true_hists.update({f+"_transf": None for f in self.true_hists})
        self.fake_hists = {}
        self.fake_hists.update({f: None for f in config.ingress.edge_features})
        self.fake_hists.update({f"{f}_idx0": None for f in config.ingress.node_features})
        self.fake_hists.update({f"{f}_idx1": None for f in config.ingress.node_features})
        self.fake_hists.update({f+"_transf": None for f in self.fake_hists})
        self.pngdir = f"{config.basedir}/{config.name}/plots"
        self.plot_labels = config.ingress.get("plot_labels", {})

    def hist(self, x, bins=torch.linspace(0, 1, 101)):
        if type(x) == list:
            x = torch.tensor(x)

        counts, edges = torch.histogram(
            x.clamp(min=0.5*(bins[0] + bins[1]), max=0.5*(bins[-2] + bins[-1])), # clip values s.t. we have under/overflow bins
            bins=bins
        )
        return [counts, edges]

    def add(self, name, feature, truth_mask):
        if self.true_hists[name] is None:
            bin_max = feature.abs().max().item()
            bin_width = round(bin_max/100) if bin_max >= 100 else bin_max/100
            # if bin_max == 0:
            #     raise ValueError(f"All values of {name} are zero for this event")
            if "layer" in name:
                self.true_hists[name] = self.hist(feature[truth_mask],  bins=torch.linspace(0, bin_max+1, int(bin_max+1)+1))
                self.fake_hists[name] = self.hist(feature[~truth_mask], bins=torch.linspace(0, bin_max+1, int(bin_max+1)+1))
            elif torch.any(feature < 0):
                self.true_hists[name] = self.hist(feature[truth_mask],  bins=torch.linspace(-100*bin_width, 100*bin_width, 51))
                self.fake_hists[name] = self.hist(feature[~truth_mask], bins=torch.linspace(-100*bin_width, 100*bin_width, 51))
            else:
                self.true_hists[name] = self.hist(feature[truth_mask],  bins=torch.linspace(0, 100*bin_width, 51))
                self.fake_hists[name] = self.hist(feature[~truth_mask], bins=torch.linspace(0, 100*bin_width, 51))
        else:
            self.true_hists[name][0] += self.hist(feature[truth_mask],  bins=self.true_hists[name][1])[0]
            self.fake_hists[name][0] += self.hist(feature[~truth_mask], bins=self.fake_hists[name][1])[0]

    def plot(self, name, tag=None):
        fig, axes = plt.subplots(figsize=(10, 10))

        # Plotting
        true_counts, edges = self.true_hists[name]
        fake_counts, edges = self.fake_hists[name]
        true_mask = (true_counts > 0)
        fake_mask = (fake_counts > 0)
        centers = 0.5*(edges[1:] + edges[:-1])
        axes.hist(
            centers[true_mask], bins=edges, weights=true_counts[true_mask]/torch.sum(true_counts[true_mask]), 
            label=f"true [{int(true_counts.sum().round())}]", histtype="step", color="r"
        )
        axes.hist(
            centers[fake_mask], bins=edges, weights=fake_counts[fake_mask]/torch.sum(fake_counts[fake_mask]), 
            label=f"fake [{int(fake_counts.sum().round())}]", color="k", alpha=0.25
        )
        axes.legend(fontsize=18)

        # Get x label
        feature_name = name.replace("_transf", "").replace("_idx0", "").replace("_idx1", "")
        xlabel = self.plot_labels.get(feature_name, name)
        if "idx0" in name:
            xlabel = "Inner "+xlabel
        elif "idx1" in name:
            xlabel = "Outer "+xlabel

        # Formatting
        axes.set_xlabel(xlabel, size=24)
        axes.set_ylabel("a.u.", size=24)
        axes.tick_params(axis="both", labelsize=20)
        axes.autoscale()

        # Generate png file name
        pngbase = f"{self.pngdir}/{name}.png"

        # Save regular plot
        pngfile = pngbase.replace(".png", f"_linear_{tag}.png")
        plt.savefig(pngfile, bbox_inches="tight")
        print(f"Wrote {pngfile}")

        # Save logscale plot
        axes.set_yscale("log")
        axes.autoscale()
        pngfile = pngbase.replace(".png", f"_logscale_{tag}.png")
        plt.savefig(pngfile, bbox_inches="tight")
        print(f"Wrote {pngfile}")

        plt.close()

    def plot_all(self, tag=None):
        os.makedirs(self.pngdir, exist_ok=True)
        for name in self.true_hists:
            self.plot(name, tag=tag)

def transform(feature, transf):
    if transf == None:
        return feature
    elif transf == "rescale":
        return (feature - feature.min())/(feature.max() - feature.min())
    elif transf == "log":
        return torch.log(feature)
    elif transf == "log2":
        return torch.log2(feature)
    elif transf == "log10":
        return torch.log10(feature)
    else:
        raise ValueError(f"transformation '{transf}' not supported")

def ingress(set_name, config, save=True, plot=True):

    plots = Plots(config)

    start, stop = config.ingress.get(f"{set_name}_entry_range")
    transforms = config.ingress.get("transforms", {})
    tree = uproot.open(f"{config.ingress.input_file}:{config.ingress.ttree_name}")
    print(f"Loaded input file, reading {start} to {stop} for {set_name} set")
    data_list = []
    for batch in SimpleProgress(tree.iterate(step_size=1, filter_name=config.ingress.get("branch_filter", None), entry_start=start, entry_stop=stop)):

        batch = batch[0,:] # only one event per batch

        # Get truth labels
        truth = torch.tensor(~(batch[config.ingress.truth_label].to_numpy().astype(bool)), dtype=torch.float)
        truth_mask = truth.to(torch.bool)

        # Get indices of nodes connected by each edge
        edge_idxs = torch.tensor([batch[n].to_list() for n in config.ingress.edge_indices], dtype=torch.long)

        # Get edge features
        edge_attr = []
        for branch_name in config.ingress.edge_features:
            feature = torch.tensor(batch[branch_name].to_list(), dtype=torch.float)
            feature[torch.isinf(feature)] = feature[~torch.isinf(feature)].max()
            plots.add(branch_name, feature, truth_mask)
            feature = transform(feature, transforms.get(branch_name, None))
            plots.add(f"{branch_name}_transf", feature, truth_mask)
            edge_attr.append(feature)

        edge_attr = torch.transpose(torch.stack(edge_attr), 0, 1)

        # Get node features
        node_attr = []
        for branch_name in config.ingress.node_features:
            feature = torch.tensor(batch[branch_name].to_list(), dtype=torch.float)
            plots.add(f"{branch_name}_idx0", feature[edge_idxs[0]], truth_mask)
            plots.add(f"{branch_name}_idx1", feature[edge_idxs[1]], truth_mask)
            feature = transform(feature, transforms.get(branch_name, None))
            plots.add(f"{branch_name}_idx0_transf", feature[edge_idxs[0]], truth_mask)
            plots.add(f"{branch_name}_idx1_transf", feature[edge_idxs[1]], truth_mask)
            node_attr.append(feature)
            
        node_attr = torch.transpose(torch.stack(node_attr), 0, 1)

        if config.ingress.get("undirected", False):
            edge_idxs_bi, edge_attr_bi = to_undirected(edge_idxs, edge_attr)
            _, truth_bi = to_undirected(edge_idxs, truth)
            data = Data(x=node_attr, y=truth_bi, edge_index=edge_idxs_bi, edge_attr=edge_attr_bi)
        else:
            data = Data(x=node_attr, y=truth, edge_index=edge_idxs, edge_attr=edge_attr)

        data_list.append(data)

    if save:
        if plot:
            plots.plot_all(tag=set_name)
        outfile = f"{config.basedir}/{config.name}/datasets/{config.name}_{set_name}.pt"
        torch.save(data_list, outfile)
        print(f"Wrote {outfile}")

    return data_list

if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description="Ingress data for GNN input")
    parser.add_argument("config_json", type=str, help="config JSON")
    args = parser.parse_args()

    config = GatorConfig.from_json(args.config_json)
    # os.makedirs(config.basedir, exist_ok=True)
    os.makedirs(f"{config.basedir}/{config.name}/datasets", exist_ok=True)

    ingress("train", config)
    ingress("test", config)
    ingress("val", config)
