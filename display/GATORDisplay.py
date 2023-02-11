#!/bin/env python

import numpy as np
import plotly.graph_objs as go
import LSTDisplay as d
import math

def get_gnn_MD_list(tree, eventidx, **kwargs):
    MD_0_x = tree["MD_0_x"].array(library="pd")[eventidx];
    MD_0_y = tree["MD_0_y"].array(library="pd")[eventidx];
    MD_0_z = tree["MD_0_z"].array(library="pd")[eventidx];
    return [MD_0_x, MD_0_y, MD_0_z]

def get_gnn_LS_list(tree, eventidx, **kwargs):
    LS_MD_idx0 = tree["LS_MD_idx0"].array(library="pd")[eventidx];
    LS_MD_idx1 = tree["LS_MD_idx1"].array(library="pd")[eventidx];
    LS_isFake = tree["LS_isFake"].array(library="pd")[eventidx];
    LS_sim_pt = tree["LS_sim_pt"].array(library="pd")[eventidx];
    LS_isInTC = tree["LS_isInTrueTC"].array(library="pd")[eventidx];
    LS_score = tree["LS_score"].array(library="pd")[eventidx];
    MD_0_x = tree["MD_0_x"].array(library="pd")[eventidx];
    MD_0_y = tree["MD_0_y"].array(library="pd")[eventidx];
    MD_0_z = tree["MD_0_z"].array(library="pd")[eventidx];
    # MD_1_x = tree["MD_1_x"].array(library="pd")[eventidx];
    # MD_1_y = tree["MD_1_y"].array(library="pd")[eventidx];
    # MD_1_z = tree["MD_1_z"].array(library="pd")[eventidx];

    Xfake_gnnfail = []
    Yfake_gnnfail = []
    Zfake_gnnfail = []
    Rfake_gnnfail = []

    Xtrue_gnnfail = []
    Ytrue_gnnfail = []
    Ztrue_gnnfail = []
    Rtrue_gnnfail = []

    Xfake_gnnpass = []
    Yfake_gnnpass = []
    Zfake_gnnpass = []
    Rfake_gnnpass = []

    Xtrue_gnnpass = []
    Ytrue_gnnpass = []
    Ztrue_gnnpass = []
    Rtrue_gnnpass = []

    LS_sim_eta = tree["LS_sim_eta"].array(library="pd")[eventidx];
    LS_sim_phi = tree["LS_sim_phi"].array(library="pd")[eventidx];
    LS_sim_vx = tree["LS_sim_vx"].array(library="pd")[eventidx];
    LS_sim_vy = tree["LS_sim_vy"].array(library="pd")[eventidx];
    LS_sim_vz = tree["LS_sim_vz"].array(library="pd")[eventidx];
    LS_sim_q = tree["LS_sim_q"].array(library="pd")[eventidx];
    LS_sim_bx = tree["LS_sim_bx"].array(library="pd")[eventidx];
    LS_sim_event = tree["LS_sim_event"].array(library="pd")[eventidx];

    threshold = kwargs["threshold"]

    for iLS, (idx0, idx1, isfake, sim_pt, sim_eta, sim_phi, sim_vx, sim_vy, sim_vz, sim_q, sim_event, sim_bx, gnn) in enumerate(zip(LS_MD_idx0, LS_MD_idx1, LS_isFake, LS_sim_pt, LS_sim_eta, LS_sim_phi, LS_sim_vx, LS_sim_vy, LS_sim_vz, LS_sim_q, LS_sim_event, LS_sim_bx, LS_score)):
        if isfake:
            if gnn > threshold:
                Xfake_gnnpass += [MD_0_x[idx0], MD_0_x[idx1]] + [None] # Non is to disconnec the lines
                Yfake_gnnpass += [MD_0_y[idx0], MD_0_y[idx1]] + [None] # Non is to disconnec the lines
                Zfake_gnnpass += [MD_0_z[idx0], MD_0_z[idx1]] + [None] # Non is to disconnec the lines
                r0 = np.sqrt(MD_0_x[idx0]**2 + MD_0_y[idx0]**2)
                r1 = np.sqrt(MD_0_x[idx1]**2 + MD_0_y[idx1]**2)
                Rfake_gnnpass += [r0, r1] + [None] # Non is to disconnec the lines
            else:
                Xfake_gnnfail += [MD_0_x[idx0], MD_0_x[idx1]] + [None] # Non is to disconnec the lines
                Yfake_gnnfail += [MD_0_y[idx0], MD_0_y[idx1]] + [None] # Non is to disconnec the lines
                Zfake_gnnfail += [MD_0_z[idx0], MD_0_z[idx1]] + [None] # Non is to disconnec the lines
                r0 = np.sqrt(MD_0_x[idx0]**2 + MD_0_y[idx0]**2)
                r1 = np.sqrt(MD_0_x[idx1]**2 + MD_0_y[idx1]**2)
                Rfake_gnnfail += [r0, r1] + [None] # Non is to disconnec the lines
        else:
            if sim_pt < 0.8:
                continue
            if not d.signal_simtrk_selection(sim_pt, sim_eta, sim_phi, sim_vx, sim_vy, sim_vz, sim_q, False):
                continue
            if sim_event != 0:
                continue
            if sim_bx != 0:
                continue
            if gnn > threshold:
                Xtrue_gnnpass += [MD_0_x[idx0], MD_0_x[idx1]] + [None] # Non is to disconnec the lines
                Ytrue_gnnpass += [MD_0_y[idx0], MD_0_y[idx1]] + [None] # Non is to disconnec the lines
                Ztrue_gnnpass += [MD_0_z[idx0], MD_0_z[idx1]] + [None] # Non is to disconnec the lines
                r0 = np.sqrt(MD_0_x[idx0]**2 + MD_0_y[idx0]**2)
                r1 = np.sqrt(MD_0_x[idx1]**2 + MD_0_y[idx1]**2)
                Rtrue_gnnpass += [r0, r1] + [None] # Non is to disconnec the lines
            else:
                Xtrue_gnnfail += [MD_0_x[idx0], MD_0_x[idx1]] + [None] # Non is to disconnec the lines
                Ytrue_gnnfail += [MD_0_y[idx0], MD_0_y[idx1]] + [None] # Non is to disconnec the lines
                Ztrue_gnnfail += [MD_0_z[idx0], MD_0_z[idx1]] + [None] # Non is to disconnec the lines
                r0 = np.sqrt(MD_0_x[idx0]**2 + MD_0_y[idx0]**2)
                r1 = np.sqrt(MD_0_x[idx1]**2 + MD_0_y[idx1]**2)
                Rtrue_gnnfail += [r0, r1] + [None] # Non is to disconnec the lines

    return [Xfake_gnnfail,
            Yfake_gnnfail,
            Zfake_gnnfail,
            Rfake_gnnfail,
            Xtrue_gnnfail,
            Ytrue_gnnfail,
            Ztrue_gnnfail,
            Rtrue_gnnfail,
            Xfake_gnnpass,
            Yfake_gnnpass,
            Zfake_gnnpass,
            Rfake_gnnpass,
            Xtrue_gnnpass,
            Ytrue_gnnpass,
            Ztrue_gnnpass,
            Rtrue_gnnpass]

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

def get_gnn_MD_go_2D(tree, eventidx, **kwargs):
    x, y, z = get_gnn_MD_list(tree, eventidx)
    mds_to_draw = go.Scatter(x=x,
                             y=y,
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

def get_gnn_MD_go_RZ(tree, eventidx, **kwargs):
    x, y, z = get_gnn_MD_list(tree, eventidx)
    r = np.sqrt(x**2 + y**2)
    mds_to_draw = go.Scatter(x=z,
                             y=r,
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

def get_gnn_LS_go(tree, eventidx, **kwargs):
    Xfake_gnnfail, Yfake_gnnfail, Zfake_gnnfail, Rfake_gnnfail, Xtrue_gnnfail, Ytrue_gnnfail, Ztrue_gnnfail, Rtrue_gnnfail, Xfake_gnnpass, Yfake_gnnpass, Zfake_gnnpass, Rfake_gnnpass, Xtrue_gnnpass, Ytrue_gnnpass, Ztrue_gnnpass, Rtrue_gnnpass = get_gnn_LS_list(tree, eventidx, **kwargs)

    objs_fake_gnnfail = go.Scatter3d(
            x = Zfake_gnnfail,
            y = Yfake_gnnfail,
            z = Xfake_gnnfail,
            mode='lines',
            line=dict(
                color=kwargs["color_fake_fail"],
                width=0.5,
            ),
            opacity=kwargs["opacity_fake_fail"],
            hoverinfo='none',
            showlegend=True,
    )
    objs_fake_gnnpass = go.Scatter3d(
            x = Zfake_gnnpass,
            y = Yfake_gnnpass,
            z = Xfake_gnnpass,
            mode='lines',
            line=dict(
                color=kwargs["color_fake_pass"],
                width=0.5,
            ),
            opacity=kwargs["opacity_fake_pass"],
            hoverinfo='none',
            showlegend=True,
    )
    objs_true_gnnfail = go.Scatter3d(
            x = Ztrue_gnnfail,
            y = Ytrue_gnnfail,
            z = Xtrue_gnnfail,
            mode='lines',
            line=dict(
                color=kwargs["color_true_fail"],
                width=0.5,
            ),
            opacity=kwargs["opacity_true_fail"],
            hoverinfo='none',
            showlegend=True,
    )
    objs_true_gnnpass = go.Scatter3d(
            x = Ztrue_gnnpass,
            y = Ytrue_gnnpass,
            z = Xtrue_gnnpass,
            mode='lines',
            line=dict(
                color=kwargs["color_true_pass"],
                width=0.5,
            ),
            opacity=kwargs["opacity_true_fail"],
            hoverinfo='none',
            showlegend=True,
    )
    return [objs_fake_gnnfail, objs_fake_gnnpass, objs_true_gnnfail, objs_true_gnnpass]

def get_gnn_LS_go_2D(tree, eventidx, **kwargs):
    Xfake_gnnfail, Yfake_gnnfail, Zfake_gnnfail, Rfake_gnnfail, Xtrue_gnnfail, Ytrue_gnnfail, Ztrue_gnnfail, Rtrue_gnnfail, Xfake_gnnpass, Yfake_gnnpass, Zfake_gnnpass, Rfake_gnnpass, Xtrue_gnnpass, Ytrue_gnnpass, Ztrue_gnnpass, Rtrue_gnnpass = get_gnn_LS_list(tree, eventidx, **kwargs)

    objs_fake_gnnfail = go.Scatter(
            y = Yfake_gnnfail,
            x = Xfake_gnnfail,
            mode='lines',
            line=dict(
                color=kwargs["color_fake_fail"],
                width=0.5,
            ),
            opacity=kwargs["opacity_fake_fail"],
            hoverinfo='none',
            showlegend=True,
    )
    objs_fake_gnnpass = go.Scatter(
            y = Yfake_gnnpass,
            x = Xfake_gnnpass,
            mode='lines',
            line=dict(
                color=kwargs["color_fake_pass"],
                width=0.5,
            ),
            opacity=kwargs["opacity_fake_pass"],
            hoverinfo='none',
            showlegend=True,
    )
    objs_true_gnnfail = go.Scatter(
            y = Ytrue_gnnfail,
            x = Xtrue_gnnfail,
            mode='lines',
            line=dict(
                color=kwargs["color_true_fail"],
                width=0.5,
            ),
            opacity=kwargs["opacity_true_fail"],
            hoverinfo='none',
            showlegend=True,
    )
    objs_true_gnnpass = go.Scatter(
            y = Ytrue_gnnpass,
            x = Xtrue_gnnpass,
            mode='lines',
            line=dict(
                color=kwargs["color_true_pass"],
                width=0.5,
            ),
            opacity=kwargs["opacity_true_fail"],
            hoverinfo='none',
            showlegend=True,
    )
    return [objs_fake_gnnfail, objs_fake_gnnpass, objs_true_gnnfail, objs_true_gnnpass]

def get_gnn_LS_go_RZ(tree, eventidx, **kwargs):
    Xfake_gnnfail, Yfake_gnnfail, Zfake_gnnfail, Rfake_gnnfail, Xtrue_gnnfail, Ytrue_gnnfail, Ztrue_gnnfail, Rtrue_gnnfail, Xfake_gnnpass, Yfake_gnnpass, Zfake_gnnpass, Rfake_gnnpass, Xtrue_gnnpass, Ytrue_gnnpass, Ztrue_gnnpass, Rtrue_gnnpass = get_gnn_LS_list(tree, eventidx, **kwargs)

    objs_fake_gnnfail = go.Scatter(
            y = Rfake_gnnfail,
            x = Zfake_gnnfail,
            mode='lines',
            line=dict(
                color=kwargs["color_fake_fail"],
                width=0.5,
            ),
            opacity=kwargs["opacity_fake_fail"],
            hoverinfo='none',
            showlegend=True,
    )
    objs_fake_gnnpass = go.Scatter(
            y = Rfake_gnnpass,
            x = Zfake_gnnpass,
            mode='lines',
            line=dict(
                color=kwargs["color_fake_pass"],
                width=0.5,
            ),
            opacity=kwargs["opacity_fake_pass"],
            hoverinfo='none',
            showlegend=True,
    )
    objs_true_gnnfail = go.Scatter(
            y = Rtrue_gnnfail,
            x = Ztrue_gnnfail,
            mode='lines',
            line=dict(
                color=kwargs["color_true_fail"],
                width=0.5,
            ),
            opacity=kwargs["opacity_true_fail"],
            hoverinfo='none',
            showlegend=True,
    )
    objs_true_gnnpass = go.Scatter(
            y = Rtrue_gnnpass,
            x = Ztrue_gnnpass,
            mode='lines',
            line=dict(
                color=kwargs["color_true_pass"],
                width=0.5,
            ),
            opacity=kwargs["opacity_true_fail"],
            hoverinfo='none',
            showlegend=True,
    )
    return [objs_fake_gnnfail, objs_fake_gnnpass, objs_true_gnnfail, objs_true_gnnpass]
