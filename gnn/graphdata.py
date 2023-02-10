#!/bin/env python

import uproot
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np

def get_data(entry_start, entry_stop):

    tree = uproot.open("/home/p.chang/data/lst/CMSSW_12_2_0_pre2/LSTGnnNtuple_ttbar_PU200.root:tree")

    data_list = []

    for batch in tree.iterate(step_size=1, library="pd", filter_name="/(MD|LS)_*/", entry_start=entry_start, entry_stop=entry_stop):

        edge_index = torch.tensor([list(batch["LS_MD_idx0"])[0],
                                   list(batch["LS_MD_idx1"])[0]], dtype=torch.long)

        edge_attr = torch.tensor([list(batch["LS_pt"])[0],
                                  list(batch["LS_eta"])[0],
                                  list(batch["LS_phi"])[0]], dtype=torch.float)
        edge_attr = torch.transpose(edge_attr, 0, 1)

        target = [ 1. if x == 0 else 0. for x in list(batch["LS_isFake"])[0] ]
        y = torch.tensor(target, dtype=torch.float)

        x = torch.tensor([list(batch["MD_0_x"])[0],
                          list(batch["MD_0_y"])[0],
                          list(batch["MD_0_z"])[0],
                          list(batch["MD_1_x"])[0],
                          list(batch["MD_1_y"])[0],
                          list(batch["MD_1_z"])[0],
                          list(batch["MD_dphichange"])[0]], dtype=torch.float)
        x = torch.transpose(x, 0, 1)

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

        data_list.append(data)

    print(data_list)
    return data_list

if __name__ == "__main__":

    data = get_data(0, 95)
    torch.save(data, "LSTGnnGraph_ttbar_PU200_train.pt") # /home/p.chang/data/lst/GATOR/CMSSW_12_2_0_pre2/LSTGnnGraph_ttbar_PU200_train.pt

    data = get_data(95, 100)
    torch.save(data, "LSTGnnGraph_ttbar_PU200_test.pt") # /home/p.chang/data/lst/GATOR/CMSSW_12_2_0_pre2/LSTGnnGraph_ttbar_PU200_test.pt

    data = get_data(100, 105)
    torch.save(data, "LSTGnnGraph_ttbar_PU200_valid.pt") # /home/p.chang/data/lst/GATOR/CMSSW_12_2_0_pre2/LSTGnnGraph_ttbar_PU200_valid.pt

