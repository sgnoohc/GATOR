#!/bin/env python

import uproot
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np

def get_data(entry_start, entry_stop):

    tree = uproot.open("~/public_html/dump/forGNN/debug.root:tree")
    # tree = uproot.open("/home/users/phchang/work/gator/TrackLooper/muon.root:tree")

    data_list = []

    for batch in tree.iterate(step_size=1, library="pd", filter_name="/(MD|LS)_*/", entry_start=entry_start, entry_stop=entry_stop):
        df_md = batch[0]
        df_ls = batch[1]

        edge_index = torch.tensor([list(df_ls["LS_MD_idx0"]),
                                   list(df_ls["LS_MD_idx1"])], dtype=torch.long)

        edge_attr = torch.tensor([list(df_ls["LS_pt"]),
                                  list(df_ls["LS_eta"]),
                                  list(df_ls["LS_phi"])], dtype=torch.float)
        edge_attr = torch.transpose(edge_attr, 0, 1)

        target = [ 1. if x == 0 else 0. for x in list(df_ls["LS_isFake"]) ]
        # target = list(df_ls["LS_pt"])

        y = torch.tensor(target, dtype=torch.float)

        print(y)

        x = torch.tensor([list(df_md["MD_0_x"]),
                          list(df_md["MD_0_y"]),
                          list(df_md["MD_0_z"]),
                          list(df_md["MD_1_x"]),
                          list(df_md["MD_1_y"]),
                          list(df_md["MD_1_z"]),
                          list(df_md["MD_dphichange"])], dtype=torch.float)
        x = torch.transpose(x, 0, 1)

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

        print(x)

        data_list.append(data)

    print(data_list)
    return data_list

if __name__ == "__main__":

    data = get_data(0, 95)
    torch.save(data, "lstdata_95evts_train.pt")

    data = get_data(95, 100)
    torch.save(data, "lstdata_5evts_test.pt")

    data = get_data(100, 105)
    torch.save(data, "lstdata_5evts_valid.pt")

