#!/bin/env python

import json
import uproot
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected
import numpy as np
import os
import getpass

from configs import GatorConfig

def ingress(entry_start, entry_stop, config):

    tree = uproot.open(f"{config.ingress.input_file}:{config.ingress.ttree_name}")

    data_list = []
    for batch in tree.iterate(step_size=1, filter_name=config.ingress.branch_filter, entry_start=entry_start, entry_stop=entry_stop):

        batch = batch[0,:] # only one event per batch

        # Get indices of nodes connected by each edge
        edge_idxs = torch.tensor([batch[n].to_list() for n in config.ingress.edge_indices], dtype=torch.long)

        # Get edge features
        edge_attr = torch.tensor([batch[n].to_list() for n in config.ingress.edge_features], dtype=torch.float)
        edge_attr = torch.transpose(edge_attr, 0, 1)

        # Get node features
        node_attr = torch.tensor([batch[n].to_list() for n in config.ingress.node_features], dtype=torch.float)
        node_attr = torch.transpose(node_attr, 0, 1)

        # Get truth labels
        truth = torch.tensor(~(batch[config.ingress.truth_label].to_numpy().astype(bool)), dtype=torch.float)

        if config.ingress.undirected:
            edge_idxs_bi, edge_attr_bi = to_undirected(edge_idxs, edge_attr)
            _, truth_bi = to_undirected(edge_idxs, truth)
            data = Data(x=node_attr, y=truth_bi, edge_index=edge_idxs_bi, edge_attr=edge_attr_bi)
        else:
            data = Data(x=node_attr, y=truth, edge_index=edge_idxs, edge_attr=edge_attr)

        data_list.append(data)

    print(data_list)
    return data_list

if __name__ == "__main__":

    config = GatorConfig.from_json("configs/LS.json")
    os.makedirs(config.basedir, exist_ok=True)

    data = ingress(0, 95, config)
    torch.save(data, f"{config.basedir}/{config.name}_train.pt")
    data = ingress(95, 100, config)
    torch.save(data, f"{config.basedir}/{config.name}_test.pt")
    data = ingress(100, 105, config)
    torch.save(data, f"{config.basedir}/{config.name}_valid.pt")
