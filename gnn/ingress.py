#!/bin/env python

import os
import argparse
import uproot
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from utils import GatorConfig

def ingress(entry_start, entry_stop, config, tag, save=True):

    tree = uproot.open(f"{config.ingress.input_file}:{config.ingress.ttree_name}")

    data_list = []
    for batch in tree.iterate(step_size=1, filter_name=config.ingress.get("branch_filter", None), entry_start=entry_start, entry_stop=entry_stop):

        batch = batch[0,:] # only one event per batch

        # Get indices of nodes connected by each edge
        edge_idxs = torch.tensor([batch[n].to_list() for n in config.ingress.edge_indices], dtype=torch.long)

        # Get edge features
        edge_attr = []
        # TEMP START: temp solution for bug in LST output
        for branch_name in config.ingress.edge_features:
            feature = batch[branch_name].to_numpy()
            feature[np.isinf(feature)] = 1000
            edge_attr.append(list(feature))
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_attr = torch.transpose(edge_attr, 0, 1)
        edge_attr -= edge_attr.min(1, keepdim=True)[0]
        edge_attr /= edge_attr.max(1, keepdim=True)[0]
        # TEMP END
        # edge_attr = torch.tensor([batch[n].to_list() for n in config.ingress.edge_features], dtype=torch.float)
        # edge_attr = torch.transpose(edge_attr, 0, 1)

        # Get node features
        node_attr = torch.tensor([batch[n].to_list() for n in config.ingress.node_features], dtype=torch.float)
        node_attr = torch.transpose(node_attr, 0, 1)
        node_attr -= node_attr.min(1, keepdim=True)[0]
        node_attr /= node_attr.max(1, keepdim=True)[0]

        # Get truth labels
        truth = torch.tensor(~(batch[config.ingress.truth_label].to_numpy().astype(bool)), dtype=torch.float)

        if config.ingress.undirected:
            edge_idxs_bi, edge_attr_bi = to_undirected(edge_idxs, edge_attr)
            _, truth_bi = to_undirected(edge_idxs, truth)
            data = Data(x=node_attr, y=truth_bi, edge_index=edge_idxs_bi, edge_attr=edge_attr_bi)
        else:
            data = Data(x=node_attr, y=truth, edge_index=edge_idxs, edge_attr=edge_attr)

        data_list.append(data)

    if save:
        outfile = f"{config.basedir}/{config.name}_{tag}.pt"
        torch.save(data_list, outfile)
        print(f"Wrote {outfile}")

    return data_list

if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description="Ingress data for GNN input")
    parser.add_argument("config_json", type=str, help="config JSON")
    args = parser.parse_args()

    config = GatorConfig.from_json(args.config_json)
    os.makedirs(config.basedir, exist_ok=True)

    ingress(0, 150, config, "train")
    ingress(150, 160, config, "test")
    ingress(160, 170, config, "val")
