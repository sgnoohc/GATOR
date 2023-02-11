#!/bin/env python

import numpy as np
import plotly.graph_objs as go

def get_gnn_MD_list(tree, eventidx, **kwargs):
    MD_0_x = tree["MD_0_x"].array(library="pd")[eventidx];
    MD_0_y = tree["MD_0_y"].array(library="pd")[eventidx];
    MD_0_z = tree["MD_0_z"].array(library="pd")[eventidx];
    return [MD_0_x, MD_0_y, MD_0_z]

def get_gnn_MD_go(tree, eventidx, **kwargs):
    x, y, z = get_gnn_MD_list(tree, eventidx)
    mds_to_draw = go.Scatter3d(x=z,
                               y=y,
                               z=x,
                               mode="markers",
                               marker=dict(
                                   symbol='circle',
                                   size=kwargs["size"],
                                   color=kwargs["color"],
                                   colorscale='Viridis',
                                   opacity=kwargs["opacity"],
                                   ),
                               # text=md_idx,
                               # hoverinfo='text',
                               showlegend=True,
                              )
    return [mds_to_draw]