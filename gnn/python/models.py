#!/bin/env python

import torch
import torch_geometric
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import Sequential as SequentialGNN, MessagePassing
from torch.nn import Sequential, Linear, ReLU, Sigmoid

from utils import GatorConfig, print_title

class LeakyDNN(nn.Module):
    def __init__(self, input_size, n_hidden_layers, hidden_size):
        super().__init__()

        hidden_layers = []
        for layer_i in range(n_hidden_layers - 1):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(nn.LeakyReLU())

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            *hidden_layers,
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        print_title("Leaky DNN Config")
        print(self)

    @classmethod
    def from_config(cls, config: GatorConfig):
        n_edge_features = len(config.ingress.edge_features)
        n_node_features = len(config.ingress.node_features)
        input_size = 2*n_node_features + n_edge_features
        n_hidden_layers = config.model.get("n_hidden_layers", 1)
        hidden_size = config.model.get("hidden_size", 64)
        return cls(input_size, n_hidden_layers, hidden_size)

    def forward(self, node_attr, edge_idxs, edge_attr):
        return self.layers(torch.cat((node_attr, edge_attr), dim=1)).unsqueeze(1)

class DNN(nn.Module):
    def __init__(self, input_size, n_hidden_layers, hidden_size):
        super().__init__()

        hidden_layers = []
        for layer_i in range(n_hidden_layers - 1):
            hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            hidden_layers.append(nn.ReLU())

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        print_title("DNN Config")
        print(self)

    @classmethod
    def from_config(cls, config: GatorConfig):
        n_edge_features = len(config.ingress.edge_features)
        n_node_features = len(config.ingress.node_features)
        input_size = 2*n_node_features + n_edge_features
        n_hidden_layers = config.model.get("n_hidden_layers", 1)
        hidden_size = config.model.get("hidden_size", 64)
        return cls(input_size, n_hidden_layers, hidden_size)

    def forward(self, node_attr, edge_idxs, edge_attr):
        return self.layers(torch.cat((node_attr, edge_attr), dim=1)).unsqueeze(1)

class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers, hidden_size):
        super().__init__()

        hidden_layers = []
        for layer_i in range(n_hidden_layers - 1):
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
        for layer_i in range(n_hidden_layers - 1):
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

class InteractionNetwork(nn.Module):
    def __init__(self, n_edge_features, n_node_features, message_size, latent_node_size, 
                 n_layers=1, mlp_n_hidden_layers=2, mlp_hidden_size=200, msg_aggr="add"):
        super().__init__()

        # Edge classification model (Latent Node, Message, Latent Node) -> (1 dim edge score)
        self.edge_classifier = RelationalModel(
            2*latent_node_size + message_size,
            1,
            mlp_n_hidden_layers,
            mlp_hidden_size
        )

        # Build GNN
        gnn_layers = []
        # First "layer" of the GNN
        gnn_layers.append((
            InteractionLayer(
                n_edge_features, n_node_features, 
                message_size, latent_node_size, 
                mlp_n_hidden_layers, mlp_hidden_size, 
                msg_aggr=msg_aggr
            ),
            InteractionLayer.get_seq_str()
        ))
        # Additional "layers" of the GNN (extra rounds of message passing)
        for layer_i in range(n_layers - 1):
            gnn_layers.append((
                InteractionLayer(
                    message_size, latent_node_size, 
                    message_size, latent_node_size, 
                    mlp_n_hidden_layers, mlp_hidden_size, 
                    msg_aggr=msg_aggr
                ),
                InteractionLayer.get_seq_str()
            ))

        # Full GNN
        self.layers = SequentialGNN("node_attr, edge_index, edge_attr", gnn_layers)

        print_title("Interaction Network Config")
        print(self)

    @classmethod
    def from_config(cls, config: GatorConfig):
        n_edge_features = len(config.ingress.edge_features)
        n_node_features = len(config.ingress.node_features)
        message_size = config.model.get("message_size", n_edge_features)
        latent_node_size = config.model.get("latent_node_size", n_node_features)
        mlp_n_hidden_layers = config.model.get("mlp_n_hidden_layers", 1)
        mlp_hidden_size = config.model.get("mlp_hidden_size", 200)

        return cls(
            n_edge_features,
            n_node_features,
            message_size,
            latent_node_size,
            n_layers=config.model.get("n_message_passing_rounds", 1),
            mlp_n_hidden_layers=mlp_n_hidden_layers,
            mlp_hidden_size=mlp_hidden_size,
            msg_aggr=config.model.get("message_aggregator", "add")
        )

    def forward(self, node_attr, edge_index, edge_attr):
        latentN_nodes, edge_index, latentN_edges = self.layers(node_attr, edge_index, edge_attr)
        latentN_graph = torch.cat(
            [
                latentN_nodes[edge_index[1]], 
                latentN_nodes[edge_index[0]], 
                latentN_edges
            ], 
            dim=1
        )
        return torch.sigmoid(self.edge_classifier(latentN_graph))

class InteractionLayer(MessagePassing):
    def __init__(self, n_edge_features, n_node_features, message_size, latent_node_size, 
                 mlp_n_hidden_layers, mlp_hidden_size, msg_aggr="add"):

        super().__init__(aggr=msg_aggr, flow="source_to_target")
        self.msg = Tensor()

        # Message calculation model (Node, Edge, Node) -> (Message)
        # i.e. "message function"
        self.message_mlp = RelationalModel(
            2*n_node_features + n_edge_features,
            message_size,
            mlp_n_hidden_layers,
            mlp_hidden_size
        )

        # Latent node calculation model (Node, Sum_i Message_i) -> (Latent Node)
        # i.e. "readout function"
        self.readout_mlp = ObjectModel(
            n_node_features + message_size,
            latent_node_size,
            mlp_n_hidden_layers,
            mlp_hidden_size
        )

    @staticmethod
    def get_seq_str():
        return "node_attr, edge_index, edge_attr -> node_attr, edge_index, edge_attr"

    def forward(self, node_attr, edge_index, edge_attr):
        latent_node_attr = self.propagate(edge_index, x=node_attr, edge_attr=edge_attr, size=None)
        return latent_node_attr, edge_index, self.msg

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

class GCNConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.msg = Tensor()

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out, self.msg

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        self.msg = norm.view(-1, 1) * x_j
        return self.msg
