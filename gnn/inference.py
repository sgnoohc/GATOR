#!/bin/env python

import os
import getpass
import sys
import argparse
from time import time

import numpy as np
from torch import optim
from torch.optim.lr_scheduler import StepLR

import uproot
import torch
import torch_geometric
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid

from model import InteractionNetwork

device = "cuda"

model = "/blue/p.chang/{}/trained_models/train_hiddensize200_PyG_LST_epoch50_lr0.005_0.8GeV_redo.pt".format(getpass.getuser())
# Parsing the setting from the file name of the model
hiddensize = int(model.rsplit("hiddensize")[1].split("_")[0])
epoch = int(model.rsplit("epoch")[1].split("_")[0])
lr = float(model.rsplit("lr")[1].split("_")[0])

interaction_network = InteractionNetwork(hiddensize)
interaction_network.load_state_dict(torch.load(model))

interaction_network.eval()

test_loader = torch.load("/blue/p.chang/p.chang/data/lst/GATOR/CMSSW_12_2_0_pre2/LSTGnnUndirGraph_ttbar_PU200_test.pt")

os.system("mkdir -p /blue/p.chang/{}/data/lst/GATOR/inference".format(getpass.getuser()))
csv = "/blue/p.chang/{}/data/lst/GATOR/inference/data_hiddensize{}_lr{}_epoch{}.csv".format(getpass.getuser(), hiddensize, lr, epoch)
f_csv = open(csv, "w")

for idx, data in enumerate(test_loader):
    data = data.to(device)
    interaction_network = interaction_network.to(device)

    # # Uncomment to time the inference
    # times = []
    # for i in range(100):
    #     start = time()
    #     output = interaction_network(data.x, data.edge_index, data.edge_attr)
    #     end = time()
    #     if i > 10:
    #         times.append(end - start)
    #     # print("end - start", end - start)
    # print("avg", np.sum(times) / len(times))

    output = interaction_network(data.x, data.edge_index, data.edge_attr)

    for y, o in zip(data.y, output):
        f_csv.write("{},{},{}\n".format(idx, float(y), float(o)))
