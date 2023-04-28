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

SUPPORTED_ACTIVATIONS = [
    nn.ReLU,
    nn.Sigmoid
]

ACTIVATIONFUNC = """
"""
ACTIVATIONFUNC = "\n".join(ACTIVATIONFUNC.split("\n")[1:-1])

CUDAMATMULFUNC = """
enum ActivationFunctions
{
    ReLU,
    Sigmoid,
    None
};

__global__ void neuralNetworkLayer(float *A, float *B, float* C, float* bias, 
                                   int A_cols, int C_rows, int C_cols, ActivationFunctions activation)
{
    /**
     * Computes the result (matrix C) of a single layer of a neural network:
     * C = activation(AB + bias)
     * where B is the matrix of weights and A is the input vector. This code is stolen from
     * here: https://github.com/charitha22/workspace/blob/master/cuda/mm/naive_matrix_multiply.cu
     */
    int row = blockIdx.y*blockDim.y + threadIdx.y;   
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < C_rows && col < C_cols)
    {
        float value = 0.;
        for (int inner = 0; inner < A_cols; ++inner)
        {
            value += A[row*A_cols + inner]*B[inner*C_cols + col];
        }
        value += bias[row*C_cols + col];
        switch (activation)
        {
        case (ReLU):
            C[row*C_cols + col] = (value > 0.) ? value : 0.;
            break;
        case (Sigmoid):
            C[row*C_cols + col] = exp(value)/(exp(value) + 1);
            break;
        default:
            C[row*C_cols + col] = value;
            break;
        }
    }
}
"""
CUDAMATMULFUNC = "\n".join(CUDAMATMULFUNC.split("\n")[1:-1])

CUDAMATINIT="""
// Initialize {matrix} and allocate device memory for it
int {matrix}_rows = {N}, {matrix}_cols = {M};
int {matrix}_size = {matrix}_rows*{matrix}_cols;
float* gpu_{matrix};
cudaMalloc(&gpu_{matrix}, {matrix}_size*sizeof(float));
"""
CUDAMATINIT = "\n".join(CUDAMATINIT.split("\n")[1:-1])

CUDAMAT2GPU = "cudaMemcpy(gpu_{matrix}, {matrix}, {matrix}_size*sizeof(float), cudaMemcpyHostToDevice);"
CUDAMAT2HOST = "cudaMemcpy({matrix}, gpu_{matrix}, {matrix}_size*sizeof(float), cudaMemcpyDeviceToHost);"

CUDAMATMUL="""
dim3 {C}_block({C}_cols, {C}_rows, 1);
dim3 {C}_grid(({C}_cols + {C}_block.x - 1)/{C}_block.x, ({C}_rows + {C}_block.y - 1)/{C}_block.y);
neuralNetworkLayer<<<{C}_grid, {C}_block>>>(gpu_{A}, gpu_{B}, gpu_{C}, gpu_{bias}, {A}_cols, {C}_rows, {C}_cols, {activation});
// Wait for GPU to finish before accessing on host
cudaDeviceSynchronize();
"""
CUDAMATMUL = "\n".join(CUDAMATMUL.split("\n")[1:-1])

MATMUL = """
float {result}[{N2}];
for (unsigned int col = 0; col < {N2}; ++col)
{{
    {result}[col] = 0;
    for (unsigned int inner = 0; inner < {M}; ++inner)
    {{
        {result}[col] += {matrix1}[inner]*{matrix2}[inner][col];
    }}
    {result}[col] += {bias}[col];
}}
"""
MATMUL = "\n".join(MATMUL.split("\n")[1:-1])

RELU = """
float {new}[{N2}];
for (unsigned int col = 0; col < {N2}; ++col)
{{
    {new}[col] = std::max(float(0), {old}[col]);
}}
"""
RELU = "\n".join(RELU.split("\n")[1:-1])

SIGMOID = """
float {new}[{N2}];
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

def vector_to_cpp(name, vector, init=True):
    cpp = Cpp()
    vec = [fmt(val) for val in vector.tolist()]
    if init:
        cpp.add(f"float {name}[{len(vec)}] = {{")
    else:
        cpp.add(f"{name} = {{")

    cpp.indent()
    cpp.add(",".join(vec))
    cpp.dedent()
    cpp.add("};")

    return cpp

def matrix_to_cpp(name, matrix, flat=False, init=True):
    n_x, n_y = matrix.size()
    cpp = Cpp()
    if init:
        if flat:
            cpp.add(f"float {name}[{n_x}*{n_y}] = {{")
        else:
            cpp.add(f"float {name}[{n_x}][{n_y}] = {{")
    else:
        cpp.add(f"{name} = {{")
    cpp.indent()

    for row_i in range(n_x):
        row = [fmt(val) for val in matrix[row_i].tolist()]
        if flat:
            if row_i == n_x - 1:
                cpp.add(f"{','.join(row)}")
            else:
                cpp.add(f"{','.join(row)},")
        else:
            cpp.add(f"{{ {','.join(row)} }},")

    cpp.dedent()
    cpp.add("};")
    return cpp

def nn_to_cuda(config, model, name="neuralNetwork"):
    n_edge_features = len(config.ingress.edge_features)
    n_node_features = len(config.ingress.node_features)
    input_size = 2*n_node_features + n_edge_features

    cpp_header = Cpp()
    cpp_header.add(f"float {name}(float* x);")

    cpp = Cpp()
    cpp.add("#include <math.h>")
    cpp.comment("CUDA libraries")
    cpp.add("#include <cuda.h>")
    cpp.add("#include <cuda_runtime.h>")
    cpp.comment("Include associated header file")
    cpp.add("#include \"T5_NN.cuh\"")
    cpp.newline()
    cpp.add(CUDAMATMULFUNC)
    cpp.newline()
    cpp.add(f"float {name}(float* x)")
    cpp.add("{")
    cpp.indent()
    cpp.comment([
        f"Auto-generated from the following PyTorch (v{torch.__version__}) model:",
        f"{model}",
        "",
        "Implements the calculation of the discriminant for a simple neural network",
        "with some CUDA acceleration"
    ])
    cpp.newline()

    prev_var = "x"
    N1, M = 1, input_size
    cpp.add(CUDAMATINIT.format(matrix=prev_var, N=N1, M=M))
    cpp.add(CUDAMAT2GPU.format(matrix=prev_var))
    cpp.newline()
    for layer_i, layer in enumerate(model.layers):
        if type(layer) == nn.Linear:
            this_var = f"x_{layer_i}"
            bias_var = f"bias_{layer_i}"
            wgts_var = f"wgtT_{layer_i}"
            N2, M = layer.weight.T.size()
            # Initialize output vector
            cpp.add(CUDAMATINIT.format(matrix=this_var, N=N1, M=M))
            cpp.add(f"float {this_var}[{N1*M}] = {{ 0. }};")
            cpp.add(CUDAMAT2GPU.format(matrix=this_var))
            # Initialize bias vector
            cpp.add(CUDAMATINIT.format(matrix=bias_var, N=N1, M=M))
            cpp.add(vector_to_cpp(bias_var, layer.bias).cpp)
            cpp.add(CUDAMAT2GPU.format(matrix=bias_var))
            # Initialize weights matrix
            cpp.add(CUDAMATINIT.format(matrix=wgts_var, N=N2, M=M))
            cpp.add(matrix_to_cpp(wgts_var, layer.weight.T, flat=True).cpp)
            cpp.add(CUDAMAT2GPU.format(matrix=wgts_var))
            cpp.newline()
            # Add matmul call
            cpp.comment(f"({layer_i}): {layer} => x = x*W_T + b")
            if type(model.layers[layer_i+1]) in SUPPORTED_ACTIVATIONS:
                activation = model.layers[layer_i+1].__str__()[:-2]
            else:
                activation = "None"
            cpp.add(CUDAMATMUL.format(
                A=prev_var,
                B=wgts_var,
                C=this_var,
                bias=bias_var,
                activation=activation
            ))
            # Copy results
            cpp.comment("Get results")
            cpp.add(CUDAMAT2HOST.format(matrix=this_var))
            cpp.newline()
            # Free matrices
            cpp.comment("Clean up")
            cpp.add(f"cudaFree(gpu_{prev_var});")
            cpp.add(f"cudaFree(gpu_{bias_var});")
            cpp.add(f"cudaFree(gpu_{wgts_var});")
            prev_var = this_var
            if layer_i == len(model.layers) - 2:
                cpp.add(f"cudaFree(gpu_{prev_var});")

            cpp.newline()


    cpp.add(f"return {prev_var}[0];")
    cpp.dedent()
    cpp.add("}")
    return cpp_header.render(), cpp.render()

def nn_to_cpp(config, model, name="neuralNetwork"):
    n_edge_features = len(config.ingress.edge_features)
    n_node_features = len(config.ingress.node_features)
    input_size = 2*n_node_features + n_edge_features

    cpp = Cpp()
    cpp.add("#include <iostream>")
    cpp.add("#include <math.h>")
    cpp.newline()
    cpp.add(f"float {name}(float x[{input_size}])")
    cpp.add("{")
    cpp.indent()
    cpp.comment([
        f"Auto-generated from the following PyTorch (v{torch.__version__}) model:",
        f"{model}",
        "",
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
        "",
        "Implements several tests for a simple neural network"
    ])
    cpp.newline()

    for test_i, data in enumerate(test_loader):
        if test_i > n_tests:
            break

        data = data.to(device)
        model = model.to(device)

        cpp.comment(f"Test {test_i}")
        feat_var = f"x_{test_i}"
        x = torch.cat((data.x, data.edge_attr), dim=1)
        cpp.add(vector_to_cpp(feat_var, x[0]).cpp)

        output = model(data.x, data.edge_index, data.edge_attr)
        
        cpp.add(f"std::cout << \"Test {test_i}: \" << {name}({feat_var}) << \" (obtained) \" << {output.item()} << \" (actual)\" << std::endl;")
        cpp.newline()

    cpp.dedent()
    cpp.add("}")

    cpp.newline()
    cpp.add("int main()")
    cpp.add("{")
    cpp.indent()
    cpp.add(f"{name}Tests();")
    cpp.add("return 0;")
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
    cpp_file = f"{config.basedir}/{config.name}/{config.name}.cc"
    with open(cpp_file, "w") as f:
        f.write(nn_to_cpp(config, model))
        f.write("\n\n")
        f.write(tests_to_cpp(model, test_loader, n_tests=10))
        print(f"Wrote {cpp_file}")

    cuda_file = f"{config.basedir}/{config.name}/{config.name}.cu"
    cuh, cu = nn_to_cuda(config, model)
    with open(cuda_file, "w") as f:
        f.write(cu)
        print(f"Wrote {cuda_file}")

    with open(cuda_file.replace(".cu", ".cuh"), "w") as f:
        f.write(cuh)
        print(f"Wrote {cuda_file.replace('.cu', '.cuh')}")
