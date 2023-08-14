#!/bin/env python

import os
import argparse
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import uproot
import awkward as ak
import pandas as pd
from sklearn.metrics import roc_curve

import models
from utils import GatorConfig
from datasets import EdgeDataset, EdgeDataBatch

def trim_leading_newline(multiline_str):
    return "\n".join(multiline_str.split("\n")[1:-1])

SUPPORTED_ACTIVATIONS = [
    nn.ReLU,
    nn.LeakyReLU,
    nn.Sigmoid
]

NEURALNETWORKWEIGHTS_CUH = """
#ifndef NeuralNetworkWeights_cuh
#define NeuralNetworkWeights_cuh

#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#else
#define CUDA_CONST_VAR
#endif

namespace T5DNN
{{
{matrices}
}}

#endif
"""
NEURALNETWORKWEIGHTS_CUH = trim_leading_newline(NEURALNETWORKWEIGHTS_CUH)

NEURALNETWORK_CUH = """
#ifndef NeuralNetwork_cuh
#define NeuralNetwork_cuh

#ifdef __CUDACC__
#define CUDA_HOSTDEV  __host__ __device__
#define CUDA_DEV __device__
#define CUDA_CONST_VAR __device__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_CONST_VAR
#endif

#include "Constants.cuh"
#include "NeuralNetworkWeights.cuh"
#include "EndcapGeometry.cuh"
#include "TiltedGeometry.h"
#include "Segment.cuh"
#include "MiniDoublet.cuh"
#include "Module.cuh"
#include "Hit.cuh"
#include "PrintUtil.h"
#include "Triplet.cuh"

namespace T5DNN
{{
{working_points}

    CUDA_DEV float runInference(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, 
                                struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, 
                                float* xVec, float* yVec, unsigned int* mdIndices, const uint16_t* lowerModuleIndices, 
                                unsigned int& innerTripletIndex, unsigned int& outerTripletIndex, 
                                float& innerRadius, float& outerRadius, float& bridgeRadius);
}}
#endif
"""
NEURALNETWORK_CUH = trim_leading_newline(NEURALNETWORK_CUH)

NEURALNETWORK_CU = """
#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#endif
#include "NeuralNetwork.cuh"

__device__ float T5DNN::runInference(struct SDL::modules& modulesInGPU, struct SDL::miniDoublets& mdsInGPU, 
                                     struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, 
                                     float* xVec, float* yVec, unsigned int* mdIndices, const uint16_t* lowerModuleIndices, 
                                     unsigned int& innerTripletIndex, unsigned int& outerTripletIndex, 
                                     float& innerRadius, float& outerRadius, float& bridgeRadius)
{{
    // Unpack x-coordinates of hits
    float x1 = xVec[0];
    float x2 = xVec[1];
    float x3 = xVec[2];
    float x4 = xVec[3];
    float x5 = xVec[4];
    // Unpack y-coordinates of hits
    float y1 = yVec[0];
    float y2 = yVec[1];
    float y3 = yVec[2];
    float y4 = yVec[3];
    float y5 = yVec[4];
    // Unpack module indices
    unsigned int mdIndex1 = mdIndices[0];
    unsigned int mdIndex2 = mdIndices[1];
    unsigned int mdIndex3 = mdIndices[2];
    unsigned int mdIndex4 = mdIndices[3];
    unsigned int mdIndex5 = mdIndices[4];
    // Unpack module indices
    uint16_t lowerModuleIndex1 = lowerModuleIndices[0];
    uint16_t lowerModuleIndex2 = lowerModuleIndices[1];
    uint16_t lowerModuleIndex3 = lowerModuleIndices[2];
    uint16_t lowerModuleIndex4 = lowerModuleIndices[3];
    uint16_t lowerModuleIndex5 = lowerModuleIndices[4];

    // Compute some convenience variables
    short layer2_adjustment = 0;
    if (modulesInGPU.layers[lowerModuleIndex1] == 1)
    {{
        layer2_adjustment = 1; // get upper segment to be in second layer
    }}
    unsigned int md_idx_for_t5_eta_phi = segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTripletIndex + layer2_adjustment]];
    bool is_endcap1 = (modulesInGPU.subdets[lowerModuleIndex1] == 4);                // true if anchor hit 1 is in the endcap
    bool is_endcap2 = (modulesInGPU.subdets[lowerModuleIndex2] == 4);                // true if anchor hit 2 is in the endcap
    bool is_endcap3 = (modulesInGPU.subdets[lowerModuleIndex3] == 4);                // true if anchor hit 3 is in the endcap
    bool is_endcap4 = (modulesInGPU.subdets[lowerModuleIndex4] == 4);                // true if anchor hit 4 is in the endcap
    bool is_endcap5 = (modulesInGPU.subdets[lowerModuleIndex5] == 4);                // true if anchor hit 5 is in the endcap

{features_cpp}

{neural_network_cpp}
    return {output_layer}[0];
}}
"""
NEURALNETWORK_CU = trim_leading_newline(NEURALNETWORK_CU)

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
MATMUL = trim_leading_newline(MATMUL)

RELU = """
float {new}[{N2}];
for (unsigned int col = 0; col < {N2}; ++col)
{{
    {new}[col] = ({old}[col] > 0.f) ? {old}[col] : 0.f;
}}
"""
RELU = trim_leading_newline(RELU)

LEAKYRELU = """
float {new}[{N2}];
for (unsigned int col = 0; col < {N2}; ++col)
{{
    {new}[col] = ({old}[col] > 0.f) ? {old}[col] : {slope}f*{old}[col];
}}
"""
LEAKYRELU = trim_leading_newline(LEAKYRELU)

SIGMOID = """
float {new}[{N2}];
for (unsigned int col = 0; col < {N2}; ++col)
{{
    {new}[col] = exp({old}[col])/(exp({old}[col]) + 1);
}}
"""
SIGMOID = trim_leading_newline(SIGMOID)

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
    decimals = 7
    return f"{num:>{decimals+3}.{decimals}f}f"

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

def nn_to_cpp(config, model, namespace="T5DNN"):
    n_edge_features = len(config.ingress.edge_features)
    n_node_features = len(config.ingress.node_features)
    input_size = 2*n_node_features + n_edge_features

    matmul_cpp = Cpp()
    matrix_cpp = Cpp()
    matmul_cpp.indent()
    matrix_cpp.indent()

    prev_arr = "x"
    N1, M = 1, input_size
    for layer_i, layer in enumerate(model.layers):
        if type(layer) == nn.Linear:
            # x = torch.matmul(x, layer.weight.T) + layer.bias
            matmul_cpp.comment(f"({layer_i}): {layer} => x = x*W_T + b")
            this_arr = f"x_{layer_i}"
            bias_arr = f"bias_{layer_i}"
            matrix_cpp.add(vector_to_cpp(bias_arr, layer.bias).cpp)
            wgts_arr = f"wgtT_{layer_i}"
            matrix_cpp.add(matrix_to_cpp(wgts_arr, layer.weight.T).cpp)
            M, N2 = layer.weight.T.size()

            # add namespace
            bias_arr = f"{namespace}::{bias_arr}"
            wgts_arr = f"{namespace}::{wgts_arr}"

            matmul_cpp.add(MATMUL.format(
                result=this_arr,
                matrix1=prev_arr,
                matrix2=wgts_arr,
                bias=bias_arr,
                N1=N1,
                N2=N2,
                M=M
            ))
            prev_arr = this_arr
            matmul_cpp.newline()
        elif type(layer) in SUPPORTED_ACTIVATIONS:
            if type(layer) == nn.Sigmoid:
                matmul_cpp.comment(f"({layer_i}): {layer}")
                this_arr = f"x_{layer_i}"
                matmul_cpp.add(SIGMOID.format(
                    new=this_arr,
                    old=prev_arr,
                    N1=N1,
                    N2=N2
                ))
                prev_arr = this_arr
                matmul_cpp.newline()
            elif type(layer) == nn.ReLU:
                matmul_cpp.comment(f"({layer_i}): {layer}")
                this_arr = f"x_{layer_i}"
                matmul_cpp.add(RELU.format(
                    new=this_arr,
                    old=prev_arr,
                    N1=N1,
                    N2=N2
                ))
                prev_arr = this_arr
                matmul_cpp.newline()
            elif type(layer) == nn.LeakyReLU:
                matmul_cpp.comment(f"({layer_i}): {layer}")
                this_arr = f"x_{layer_i}"
                matmul_cpp.add(LEAKYRELU.format(
                    new=this_arr,
                    old=prev_arr,
                    N1=N1,
                    N2=N2,
                    slope=0.01
                ))
                prev_arr = this_arr
                matmul_cpp.newline()
        else:
            raise Exception(
                f"{layer} is not a Linear layer, nor is it in the list of supported activations"
            )

    return matmul_cpp, matrix_cpp, prev_arr

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

    saved_model = config.get_outfile(subdir="models", epoch=args.epoch)
    Model = getattr(models, config.model.name)
    model = Model(config).to(device)
    model.load_state_dict(torch.load(saved_model, map_location=device))
    model.eval()

    # Generate matrix multiplcation/declaration C++ code
    matmul_cpp, matrix_cpp, output_layer = nn_to_cpp(config, model)

    # Write features vector template
    features_cpp = Cpp()
    features_cpp.indent()
    n_features = 2*len(config.ingress.node_features) + len(config.ingress.edge_features)
    features_cpp.comment("Build DNN input vector (corresponding output N-tuple branch noted in parenthetical in comment)")
    features_cpp.add(f"float x[{n_features}] = {{")
    features_cpp.indent()
    for node in ["inner", "outer"]:
        for feature in config.ingress.node_features:
            features_cpp.add(f"-1.f, // FIXME: {node} {feature} ({config.ingress.transforms.get(feature, 'no')} transf)")
    for feature in config.ingress.edge_features:
        features_cpp.add(f"-1.f, // FIXME: {feature} ({config.ingress.transforms.get(feature, 'no')} transf)")
    features_cpp.dedent()
    features_cpp.add("};")

    # Get ingredients for LST FPR and TPR
    with uproot.open("/blue/p.chang/jguiang/data/lst/GATOR/CMSSW_12_2_0_pre2/LSTGnnNtuple.root") as f:
        is_fake = ak.values_astype(f["tree"]["t5_isFake"].array(), "bool")
        lst_FP = ak.sum(is_fake)                # number of false positives (fake T5s produced by LST)
        lst_TP = ak.sum(~is_fake)               # number of true positives (real T5s produced by LST)
    with uproot.open(config.ingress.input_file) as f:
        is_fake = ak.values_astype(f["tree"]["t5_isFake"].array(), "bool")
        lst_N = ak.sum(is_fake)                 # total number of fakes
        lst_P = ak.sum(~is_fake)                # total number of trues

    lst_fpr = lst_FP/lst_N
    lst_tpr = lst_TP/lst_P

    # Get DNN ROC curve
    test_df = pd.read_csv(saved_model.replace("models", "inferences").replace("_model.pt", "_test.csv"))
    fpr, tpr, thresh = roc_curve(test_df.truth, test_df.score)


    # Write some working points
    wps_cpp = Cpp()
    wps_cpp.indent()
    wps_cpp.comment(f"Working points matching LST fake rate ({lst_fpr*100:.1f}%) or signal acceptance ({lst_tpr*100:.1f}%)")
    wps_cpp.add(f"CUDA_CONST_VAR const float LSTWP1 = {thresh[fpr >= lst_fpr][0]:.7f}f; // {tpr[fpr >= lst_fpr][0]*100:>4.1f}% TPR, {lst_fpr*100:>4.1f}% FPR")
    wps_cpp.add(f"CUDA_CONST_VAR const float LSTWP2 = {thresh[tpr >= lst_tpr][0]:.7f}f; // {lst_tpr*100:>4.1f}% TPR, {fpr[tpr >= lst_tpr][0]*100:>4.1f}% FPR")
    wps_cpp.comment("Other working points")
    wps_cpp.add(f"CUDA_CONST_VAR const float WP70   = {thresh[tpr >= 0.700][0]:.7f}f; // 70.0% TPR, {fpr[tpr >= 0.700][0]*100:>4.1f}% FPR")
    wps_cpp.add(f"CUDA_CONST_VAR const float WP75   = {thresh[tpr >= 0.750][0]:.7f}f; // 75.0% TPR, {fpr[tpr >= 0.750][0]*100:>4.1f}% FPR")
    wps_cpp.add(f"CUDA_CONST_VAR const float WP80   = {thresh[tpr >= 0.800][0]:.7f}f; // 80.0% TPR, {fpr[tpr >= 0.800][0]*100:>4.1f}% FPR")
    wps_cpp.add(f"CUDA_CONST_VAR const float WP85   = {thresh[tpr >= 0.850][0]:.7f}f; // 85.0% TPR, {fpr[tpr >= 0.850][0]*100:>4.1f}% FPR")
    wps_cpp.add(f"CUDA_CONST_VAR const float WP90   = {thresh[tpr >= 0.900][0]:.7f}f; // 90.0% TPR, {fpr[tpr >= 0.900][0]*100:>4.1f}% FPR")
    wps_cpp.add(f"CUDA_CONST_VAR const float WP95   = {thresh[tpr >= 0.950][0]:.7f}f; // 95.0% TPR, {fpr[tpr >= 0.950][0]*100:>4.1f}% FPR")
    wps_cpp.add(f"CUDA_CONST_VAR const float WP97p5 = {thresh[tpr >= 0.975][0]:.7f}f; // 97.5% TPR, {fpr[tpr >= 0.975][0]*100:>4.1f}% FPR")
    wps_cpp.add(f"CUDA_CONST_VAR const float WP99   = {thresh[tpr >= 0.990][0]:.7f}f; // 99.0% TPR, {fpr[tpr >= 0.990][0]*100:>4.1f}% FPR")
    wps_cpp.add(f"CUDA_CONST_VAR const float WP99p9 = {thresh[tpr >= 0.999][0]:.7f}f; // 99.9% TPR, {fpr[tpr >= 0.999][0]*100:>4.1f}% FPR")

    cu_file = f"{config.base_dir}/{config.name}/NeuralNetwork.cu"
    with open(cu_file, "w") as f:
        cu = NEURALNETWORK_CU.format(
            features_cpp=features_cpp.render(),
            neural_network_cpp=matmul_cpp.render(),
            output_layer=output_layer
        )
        f.write(cu)
        print(f"Wrote {cu_file}")

    cuh_file = f"{config.base_dir}/{config.name}/NeuralNetwork.cuh"
    with open(cuh_file, "w") as f:
        cuh = NEURALNETWORK_CUH.format(
            working_points=wps_cpp.render()
        )
        f.write(cuh)
        print(f"Wrote {cuh_file}")

    cuh_file = f"{config.base_dir}/{config.name}/NeuralNetworkWeights.cuh"
    with open(cuh_file, "w") as f:
        cuh = NEURALNETWORKWEIGHTS_CUH.format(
            matrices=matrix_cpp.render()
        )
        cuh = cuh.replace("float", "CUDA_CONST_VAR const float")
        f.write(cuh)
        print(f"Wrote {cuh_file}")
