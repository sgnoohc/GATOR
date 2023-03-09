#!/bin/env python

import torch
import torch_geometric
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid

from configs import GatorConfig

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
    def __init__(self, R1, O, R2):
        super(InteractionNetwork).__init__(aggr="max", flow="source_to_target")
        self.R1 = R1
        self.O = O
        self.R2 = R2
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

class ChangNet(InteractionNetwork):
    def __init__(self, config: GatorConfig):
        n_edge_features = len(config.ingress.edge_features)
        n_node_features = len(config.ingress.node_features)
        n_msg_features = config.model.get("n_message_features", 17)
        n_lnode_features = config.model.get("n_latent_node_features", n_node_features)
        n_hidden_layers = config.model.get("n_hidden_layers", 200)

        # Message Calculation Model (Node, Edge, Node) -> (Message)
        # i.e. "message function"
        R1 = RelationalModel(
            2*n_node_features + n_edge_features,
            n_msg_features,
            n_hidden_layers
        )

        # Latent Node Calculation Model (Node, Sum_i Message_i) -> (Latent Node)
        # i.e. "readout function"
        O = ObjectModel(
            n_node_features + n_msg_features,
            n_lnode_features,
            n_hidden_layers
        )

        # Edge Classification Model (Latent Node, Message, Latent Node) -> (1 dim edge score)
        R2 = RelationalModel(
            2*n_lnode_features + n_msg_features,
            1,
            n_hidden_layers
        )

        super(ChangNet).__init__(R1, O, R2)
