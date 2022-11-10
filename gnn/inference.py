#!/bin/env python

import os
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

def usage():
    print("Usage:")
    print("")
    print("  $ python {} MODEL.pt".format(sys.argv[0]))
    print("")
    print("")
    sys.exit()

if len(sys.argv) < 2:
    usage()

device = "cpu"

model = sys.argv[1]
hiddensize = int(model.rsplit("hiddensize")[1].split("_")[0])
epoch = int(model.rsplit("epoch")[1].split("_")[0])

interaction_network = InteractionNetwork(hiddensize)
interaction_network.load_state_dict(torch.load(model))

interaction_network.eval()

test_loader = torch.load("lstdata_5evts_test.pt")

csv = "../result/csvs/data_epoch{}.csv".format(epoch)
f_csv = open(csv, "w")

for idx, data in enumerate(test_loader):
    data = data.to(device)
    output = interaction_network(data.x, data.edge_index, data.edge_attr)

    for y, o in zip(data.y, output):
        f_csv.write("{},{}\n".format(float(y), float(o)))
