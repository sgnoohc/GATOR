#!/bin/env python
import os
import glob
import argparse

import matplotlib.pyplot as plt
import uproot
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
        self.png_dir = f"{config.base_dir}/{config.name}/plots/ingress"
        self.plot_labels = config.ingress.get("plot_labels", {})

    def hist(self, x, bins=torch.linspace(0, 1, 101)):
        if type(x) == list:
            x = torch.tensor(x)

        counts, errors = torch.histogram(
            x.clamp(min=0.5*(bins[0] + bins[1]), max=0.5*(bins[-2] + bins[-1])), # clip values s.t. we have under/overflow bins
            bins=bins
        )
        return [counts, errors]

    def add(self, name, feature, truth_mask):
        if self.true_hists[name] is None:
            bin_max = feature.abs().max().item()
            bin_width = round(bin_max/100) if bin_max >= 100 else bin_max/100
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
            xlabel = "Inner " + xlabel
        elif "idx1" in name:
            xlabel = "Outer " + xlabel

        # Formatting
        axes.set_xlabel(xlabel, size=24)
        axes.set_ylabel("a.u.", size=24)
        axes.tick_params(axis="both", labelsize=20)
        axes.autoscale()

        # Generate png file name
        png_base = f"{self.png_dir}/{name}.png"

        # Save regular plot
        png_file = png_base.replace(".png", f"_linear_{tag}.png")
        plt.savefig(png_file, bbox_inches="tight")
        print(f"Wrote {png_file}")

        # Save logscale plot
        axes.set_yscale("log")
        axes.autoscale()
        png_file = png_base.replace(".png", f"_logscale_{tag}.png")
        plt.savefig(png_file, bbox_inches="tight")
        print(f"Wrote {png_file}")

        plt.close()

    def plot_all(self, tag=None):
        os.makedirs(self.png_dir, exist_ok=True)
        for name in self.true_hists:
            self.plot(name, tag=tag)

def transform(feature, transf):
    if transf is None:
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

def ingress_file(config, root_file, save=True, plot=True):

    plots = Plots(config)

    print(f"Loading {root_file}...")
    transforms = config.ingress.get("transforms", {})
    branch_filter = config.ingress.get("branch_filter", None)
    root_file_name = root_file.split("/")[-1].replace(".root", "")
    graphs = []
    with uproot.open(root_file) as f:
        # Collect branches to ingress
        branches = config.ingress["node_features"] + config.ingress["edge_features"] + config.ingress["edge_indices"]
        branches.append(config.ingress["truth_label"])

        # Load TTree
        tree = f[config.ingress.ttree_name].arrays(branches)

        for event in SimpleProgress(tree):
            # Get truth labels
            truth = ~torch.tensor(event[config.ingress.truth_label], dtype=torch.bool)

            # Get indices of nodes connected by each edge
            edge0, edge1 = config.ingress.edge_indices
            edge_idxs = torch.tensor([event[edge0].to_list(), event[edge1].to_list()], dtype=torch.long)

            # Get edge features
            edge_attr = []
            for branch_name in config.ingress.edge_features:
                feature = torch.tensor(event[branch_name], dtype=torch.float)          # get feature
                feature[torch.isinf(feature)] = feature[~torch.isinf(feature)].max()   # trim infinities
                plots.add(branch_name, feature, truth)                                 # add to pre-transf plots
                feature = transform(feature, transforms.get(branch_name, None))        # apply transformation
                plots.add(f"{branch_name}_transf", feature, truth)                     # add to post-transf plots
                edge_attr.append(feature)                                              # save feature

            edge_attr = torch.transpose(torch.stack(edge_attr), 0, 1)

            # Get node features
            node_attr = []
            for branch_name in config.ingress.node_features:
                feature = torch.tensor(event[branch_name], dtype=torch.float)          # get feature
                feature[torch.isinf(feature)] = feature[~torch.isinf(feature)].max()   # trim infinities
                plots.add(f"{branch_name}_idx0", feature[edge_idxs[0]], truth)         # add inner to pre-transf plots
                plots.add(f"{branch_name}_idx1", feature[edge_idxs[1]], truth)         # add outer to pre-transf plots
                feature = transform(feature, transforms.get(branch_name, None))        # apply transformation
                plots.add(f"{branch_name}_idx0_transf", feature[edge_idxs[0]], truth)  # add inner to post-transf plots
                plots.add(f"{branch_name}_idx1_transf", feature[edge_idxs[1]], truth)  # add outer to post-transf plots
                node_attr.append(feature)                                              # save feature
                
            node_attr = torch.transpose(torch.stack(node_attr), 0, 1)

            if config.ingress.get("undirected", False):
                edge_idxs_bi, edge_attr_bi = to_undirected(edge_idxs, edge_attr)
                _, truth_bi = to_undirected(edge_idxs, truth.to(torch.float))
                graph = Data(x=node_attr, y=truth_bi, edge_index=edge_idxs_bi, edge_attr=edge_attr_bi)
            else:
                graph = Data(x=node_attr, y=truth.to(torch.float), edge_index=edge_idxs, edge_attr=edge_attr)

            graphs.append(graph)

    if save:
        outfile = config.get_outfile(subdir="inputs", tag=root_file_name, short=True)
        torch.save(graphs, outfile)
        print(f"Wrote {outfile}")
        if plot:
            print("Plotting all features...")
            plots.plot_all(tag=root_file_name)

def ingress(config, save=True, plot=True):
    input_files = config.ingress.get("input_files", None)
    if not input_files is None:
        for root_file in input_files:
            ingress_file(config, root_file, save=save, plot=plot)
        
    input_dir = config.ingress.get("input_dir", None)
    if not input_dir is None:
        for root_file in glob.glob(f"{input_dir}/*.root"):
            ingress_file(config, root_file, save=save, plot=plot)
    
    if input_files is None and input_dir is None:
        raise Exception("No input files were provided")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingress data")
    parser.add_argument("config_json", type=str, help="config JSON")
    parser.add_argument(
        "--no_plots", action="store_true",
        help="do not plot features"
    )
    args = parser.parse_args()

    config = GatorConfig.from_json(args.config_json)

    ingress(config, plot=(not args.no_plots))

    print("\nDone.\n")
