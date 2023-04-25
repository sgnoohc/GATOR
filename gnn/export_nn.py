#!/bin/env python

import os
import argparse
from time import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

import models
from utils import GatorConfig
from train import get_model_filename

MATMUL = """
float {result}[N2];
for (unsigned int col = 0; col < {N2}; ++col)
{{
    {result}[col] = 0;
    for (usngined int inner = 0; inner < {M}; ++inner)
    {{
        {result}[col] += {matrix1}[inner]*{matrix2}[inner][col];
    }}
    {result}[col] += {bias}[col];
}}
"""
MATMUL = "\n".join(MATMUL.split("\n")[1:-1])

RELU = """
float {new}[N2];
for (unsigned int col = 0; col < {N2}; ++col)
{{
    {new}[col] = max(0, {old}[col]);
}}
"""
RELU = "\n".join(RELU.split("\n")[1:-1])

SIGMOID = """
float {new}[N2];
for (unsigned int col = 0; col < {N2}; ++col)
{{
    {new}[col] = exp({old}[col])/(exp({old}[col]) + 1);
}}
"""
SIGMOID = "\n".join(SIGMOID.split("\n")[1:-1])

class Cpp:
    def __init__(self, tab="    "):
        self.cpp = []
        self.__tab = tab
        self.__n_tabs = 0

    def indent(self, n=1):
        self.__n_tabs += n

    def dedent(self, n=1):
        self.__n_tabs -= n
        self.__n_tabs = max(self.__n_tabs, 0)

    def newline(self):
        self.add("")

    def __ingress_cpp(self, cpp):
        if type(cpp) == str:
            cpp = cpp.split("\n")
        elif type(cpp) != list:
            raise ValueError("can only add a single line or list of lines")

        expand = []
        for line_i, line in enumerate(cpp):
            if "\n" in line:
                expand.append((line_i, line.split("\n")))

        for insert_i, lines in expand:
            cpp = cpp[:insert_i] + lines + cpp[insert_i+1:]

        return cpp

    def comment(self, cpp):
        cpp = self.__ingress_cpp(cpp)
        if len(cpp) == 1:
            # Single-line comment
            self.add(f"// {cpp[0]}")
        else:
            # Multi-line comment
            self.add("/**")
            for line in cpp:
                self.add(f" * {line}")
            self.add(" */")

    def add(self, cpp):
        cpp = self.__ingress_cpp(cpp)
        for line_i in range(len(cpp)):
            cpp[line_i] = self.__tab*self.__n_tabs + cpp[line_i]
        self.cpp += cpp
    
    def render(self):
        return "\n".join(self.cpp)

def fmt(num):
    decimals = 20
    return f"{num:>{decimals+3}.{decimals}f}"

def vector_to_cpp(name, vector):

    cpp = Cpp()
    vec = [fmt(val) for val in vector.tolist()]
    cpp.add(f"const float {name}[{len(vec)}] = {{ {','.join(vec)} }};")

    return cpp

def matrix_to_cpp(name, matrix):
    n_x, n_y = matrix.size()

    cpp = Cpp()
    cpp.add(f"const float {name}[{n_x}][{n_y}] = {{")
    cpp.indent()

    for row_i in range(n_x):
        row = [fmt(val) for val in matrix[row_i].tolist()]
        cpp.add(f"{{ {','.join(row)} }},")

    cpp.dedent()
    cpp.add("};")
    return cpp

def nn_to_cpp(config, model, name="neuralNetwork"):
    n_edge_features = len(config.ingress.edge_features)
    n_node_features = len(config.ingress.node_features)
    input_size = 2*n_node_features + n_edge_features

    cpp = Cpp()
    cpp.add(f"__global__ float {name}(float x[{input_size}])")
    cpp.add("{")
    cpp.indent()
    cpp.comment([
        f"Auto-generated from the following PyTorch (v{torch.__version__}) model:",
        f"{model}",
        "Implements the calculation of the discriminant for a simple neural network"
    ])
    cpp.newline()

    prev_var = "x"
    N1, M = 1, input_size
    for layer_i, layer in enumerate(model.layers):
        if type(layer) == nn.Linear:
            # x = torch.matmul(x, layer.weight.T) + layer.bias
            cpp.comment(f"({layer_i}): {layer} => x = x*W_T + b")
            this_var = f"x_{layer_i}"
            bias_var = f"bias_{layer_i}"
            cpp.add(vector_to_cpp(bias_var, layer.bias).cpp)
            wgts_var = f"wgtT_{layer_i}"
            cpp.add(matrix_to_cpp(wgts_var, layer.weight.T).cpp)
            M, N2 = layer.weight.T.size()

            cpp.add(MATMUL.format(
                result=this_var,
                matrix1=prev_var,
                matrix2=wgts_var,
                bias=bias_var,
                N1=N1,
                N2=N2,
                M=M
            ))
            prev_var = this_var
            cpp.newline()
        elif type(layer) == nn.Sigmoid:
            cpp.comment(f"({layer_i}): {layer}")
            this_var = f"x_{layer_i}"
            cpp.add(SIGMOID.format(
                new=this_var,
                old=prev_var,
                N1=N1,
                N2=N2
            ))
            prev_var = this_var
            cpp.newline()
        elif type(layer) == nn.ReLU:
            cpp.comment(f"({layer_i}): {layer}")
            this_var = f"x_{layer_i}"
            cpp.add(RELU.format(
                new=this_var,
                old=prev_var,
                N1=N1,
                N2=N2
            ))
            prev_var = this_var
            cpp.newline()

    cpp.add(f"return {prev_var}[0];")
    cpp.dedent()
    cpp.add("}")
    return cpp.render()

def tests_to_cpp(model, loader, n_tests=10, name="neuralNetwork"):
    cpp = Cpp()
    cpp.add(f"void {name}Tests()")
    cpp.add("{")
    cpp.indent()
    cpp.comment([
        f"Auto-generated from the following PyTorch (v{torch.__version__}) model:",
        f"{model}",
        "Implements several tests for a simple neural network"
    ])
    cpp.newline()

    for test_i, data in enumerate(test_loader):
        if test_i > n_tests:
            break

        cpp.comment(f"Test {test_i}")
        feat_var = f"x_{test_i}"
        x = torch.cat((data.x, data.edge_attr), dim=1)
        cpp.add(vector_to_cpp(feat_var, x[0]).cpp)

        output = model(data.x, data.edge_index, data.edge_attr)
        
        cpp.add(f"std::cout << \"Test {test_i}: \" << {name}({feat_var}) << \" (obtained) \" << {output.item()} << \" (actual)\"")
        cpp.newline()

    cpp.dedent()
    cpp.add("}")
    return cpp.render()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DNN as matrix multiplication")
    parser.add_argument("config_json", type=str, help="config JSON")
    parser.add_argument(
        "--epoch", type=int, default=50, metavar="N",
        help="training epoch of model to use for inference (default: 50)"
    )
    args = parser.parse_args()

    config = GatorConfig.from_json(args.config_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_model = f"{config.basedir}/{config.name}/models/{get_model_filename(config, args.epoch)}"
    Model = getattr(models, config.model.name)
    model = Model(config).to(device)
    model.load_state_dict(torch.load(saved_model, map_location=device))

    cpp_file = f"{config.basedir}/{config.name}/{config.name}.cu"
    with open(cpp_file, "w") as f:
        f.write(nn_to_cpp(config, model))
        print(f"Wrote {cpp_file}")

    test_loader = torch.load(f"{config.basedir}/{config.name}/datasets/{config.name}_test.pt")
    if config.model.name == "DNN":
        from datasets import EdgeDataset, EdgeDataBatch
        from torch.utils.data import DataLoader
        # Hacky PyG graph-level GNN inputs --> edge-level DNN inputs
        test_loader = DataLoader(
            EdgeDataset(test_loader), batch_size=1, shuffle=True,
            collate_fn=lambda batch: EdgeDataBatch(batch)
        )

    model.eval()
    cpp_file = f"{config.basedir}/{config.name}/{config.name}_tests.cc"
    with open(cpp_file, "w") as f:
        f.write(tests_to_cpp(model, test_loader, n_tests=10))
        print(f"Wrote {cpp_file}")
