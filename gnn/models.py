#!/bin/env python

import torch
import torch_geometric
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid

from utils import GatorConfig, print_title

class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers, hidden_size):
        super().__init__()

        hidden_layers = []
        for layer_i in range(n_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(nn.ReLU())

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, m):
        return self.layers(m)

class ObjectModel(nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers, hidden_size):
        super().__init__()

        hidden_layers = []
        for layer_i in range(n_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(nn.ReLU())

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)

class InteractionNetwork(MessagePassing):
    def __init__(self, message_mlp, readout_mlp, edge_classifier, n_msgpass_rounds=1, msg_aggr="max"):
        super().__init__(aggr=msg_aggr, flow="source_to_target")
        self.message_mlp = message_mlp
        self.readout_mlp = readout_mlp
        self.edge_classifier = edge_classifier
        self.n_msgpass_rounds = n_msgpass_rounds # number of rounds of message passing
        self.msg: Tensor = Tensor()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Run forward pass of InteractionNetwork
        """
        # Run first round of message passing
        x_latent = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        # Run additional rounds of message passing
        for round_i in range(self.n_msgpass_rounds - 1):
            x_latent = self.propagate(edge_index, x=x_latent, edge_attr=edge_attr, size=None)

        m2 = torch.cat([x_latent[edge_index[1]], x_latent[edge_index[0]], edge_attr], dim=1)

        return torch.sigmoid(self.edge_classifier(m2))

    def message(self, x_i, x_j, edge_attr):
        # Message Calculation Model
        # x_i --> incoming
        # x_j --> outgoing
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.msg = self.message_mlp(m1)
        return self.msg

    def update(self, aggr_out, x):
        # Latent Node Calculation Model
        c = torch.cat([x, aggr_out], dim=1)
        return self.readout_mlp(c)

class DNN(nn.Module):
    def __init__(self, config: GatorConfig):
        super().__init__()
        n_edge_features = len(config.ingress.edge_features)
        n_node_features = len(config.ingress.node_features)
        input_size = 2*n_node_features + n_edge_features
        n_hidden_layers = config.model.get("n_hidden_layers", 1)
        hidden_size = config.model.get("hidden_size", 64)

        hidden_layers = []
        for layer_i in range(n_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(nn.ReLU())

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        print("---- DNN Config ----")
        print(self)

    def forward(self, node_attr, edge_idxs, edge_attr):
        return self.layers(torch.cat((node_attr, edge_attr), dim=1)).unsqueeze(1)

class ChangNet(InteractionNetwork):
    def __init__(self, config: GatorConfig):
        n_edge_features = len(config.ingress.edge_features)
        n_node_features = len(config.ingress.node_features)
        n_msg_features = config.model.get("n_message_features", n_edge_features)
        n_lnode_features = config.model.get("n_latent_node_features", n_node_features)
        n_hidden_layers = config.model.get("n_hidden_layers", 1)
        hidden_size = config.model.get("hidden_size", 200)

        print_title("ChangNet Config")
        # Message calculation model (Node, Edge, Node) -> (Message)
        # i.e. "message function"
        message_mlp = RelationalModel(
            2*n_node_features + n_edge_features,
            n_msg_features,
            n_hidden_layers,
            hidden_size
        )
        print(f"Message MLP = {message_mlp}")

        # Latent node calculation model (Node, Sum_i Message_i) -> (Latent Node)
        # i.e. "readout function"
        readout_mlp = ObjectModel(
            n_node_features + n_msg_features,
            n_lnode_features,
            n_hidden_layers,
            hidden_size
        )
        print(f"Readout MLP = {readout_mlp}")

        # Edge classification model (Latent Node, Message, Latent Node) -> (1 dim edge score)
        edge_classifier = RelationalModel(
            # 2*n_lnode_features + n_msg_features,
            2*n_lnode_features + n_edge_features,
            1,
            n_hidden_layers,
            hidden_size
        )
        print(f"Edge classifier = {edge_classifier}")

        super().__init__(
            message_mlp, 
            readout_mlp, 
            edge_classifier, 
            n_msgpass_rounds=config.model.get("n_message_passing_rounds", 1),
            msg_aggr=config.model.get("message_aggregator", "max")
        )
