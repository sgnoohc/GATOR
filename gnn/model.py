#!/bin/env python

import torch
import torch_geometric
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid

verbose = False

def pprint_tensor(m, name):
    if verbose:
        print("{}.size()".format(name))
        print(m.size())
        print("{}".format(name))
        print(m)

class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, m):
        p = self.layers(m)
        pprint_tensor(p, "p")
        return self.layers(m)

class ObjectModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ObjectModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)


class InteractionNetwork(MessagePassing):
    def __init__(self, hidden_size):
        super(InteractionNetwork, self).__init__(aggr='add',
                                                 flow='source_to_target')
        self.R1 = RelationalModel(17, 3, hidden_size)
        self.O = ObjectModel(10, 3, hidden_size)
        self.R2 = RelationalModel(9, 1, hidden_size)
        self.E: Tensor = Tensor()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:

        pprint_tensor(edge_index, "edge_index")
        pprint_tensor(x, "x")
        pprint_tensor(edge_attr, "edge_attr")

        # propagate_type: (x: Tensor, edge_attr: Tensor)
        x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        pprint_tensor(x_tilde, "x_tilde")

        m2 = torch.cat([x_tilde[edge_index[1]],
                        x_tilde[edge_index[0]],
                        self.E], dim=1)
        return torch.sigmoid(self.R2(m2))
        # return self.R2(m2)

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming
        # x_j --> outgoing
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        pprint_tensor(m1, "m1")
        self.E = self.R1(m1)
        pprint_tensor(self.E, "self.E")
        return self.E

    def update(self, aggr_out, x):
        pprint_tensor(aggr_out, "aggr_out")
        c = torch.cat([x, aggr_out], dim=1)
        pprint_tensor(c, "c")
        pprint_tensor(self.O(c), "self.O(c)")
        return self.O(c)

