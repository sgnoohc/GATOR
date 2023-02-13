#!/bin/env python

import torch
import torch_geometric
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid

node_feat_len = 7
edge_feat_len = 3
message_feat_len = 17
latent_node_feat_len = 7

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
        super(InteractionNetwork, self).__init__(aggr='max',
                                                 flow='source_to_target')

        # Message Calculation Model (Node, Edge, Node) -> (Message)
        self.R1 = RelationalModel(
                2 * node_feat_len + edge_feat_len,
                message_feat_len,
                hidden_size)

        # Latent Node Calculation Model (Node, Sum_i Message_i) -> (Latent Node)
        self.O = ObjectModel(
                node_feat_len + message_feat_len,
                latent_node_feat_len,
                hidden_size)

        # Edge Classification Model (Latent Node, Message, Latent Node) -> (1 dim edge score)
        self.R2 = RelationalModel(
                2 * latent_node_feat_len + message_feat_len,
                1,
                hidden_size)

        self.E: Tensor = Tensor()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # type: (Tensor, Tensor, Tensor) -> Tensor

        # Edge Classification Model
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        x_tilde = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        m2 = torch.cat([x_tilde[edge_index[1]],
                        x_tilde[edge_index[0]],
                        self.E], dim=1)
        return torch.sigmoid(self.R2(m2))

    def message(self, x_i, x_j, edge_attr):
        # Message Calculation Model
        # x_i --> incoming
        # x_j --> outgoing
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.E = self.R1(m1)
        return self.E

    def update(self, aggr_out, x):
        # Latent Node Calculation Model
        c = torch.cat([x, aggr_out], dim=1)
        return self.O(c)

