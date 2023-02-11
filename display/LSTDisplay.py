#!/bin/env python

import uproot
import LSTMath as m
import plotly.graph_objs as go
import math
import json
import numpy as np


#########################################################################################################################
# Functions used to select tracks
#########################################################################################################################
def signal_simtrk_selection(pt, eta, phi, vx, vy, vz, q, TC_matched):
    if pt < 0.8 or abs(eta) > 4.0 or q == 0 or math.sqrt(vx**2 + vy**2) > 2.5 or abs(vz) > 30.:
        return False
    else:
        return True

def recoed_signal_simtrk_selection(pt, eta, phi, vx, vy, vz, q, TC_matched):
    if pt < 0.8 or abs(eta) > 4.0 or q == 0 or math.sqrt(vx**2 + vy**2) > 2.5 or abs(vz) > 30. or TC_matched == False:
        return False
    else:
        return True

#########################################################################################################################
# Modules
#########################################################################################################################
def get_modules(geom_file="data/CMSSW_12_2_0_pre2_geom.txt"):
    f = open(geom_file)
    j = json.load(f)
    zs=[]
    xs=[]
    ys=[]
    for detid in j:
        z_=[j[detid][0][0], j[detid][1][0], j[detid][2][0], j[detid][3][0], j[detid][0][0], None]
        x_=[j[detid][0][1], j[detid][1][1], j[detid][2][1], j[detid][3][1], j[detid][0][1], None]
        y_=[j[detid][0][2], j[detid][1][2], j[detid][2][2], j[detid][3][2], j[detid][0][2], None]
        zs += x_
        xs += z_
        ys += y_
    mods = go.Scatter3d(x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        line=dict(
                            color="rgb(0,255,255,0.5)",
                            ),
                        opacity=0.1,
                        hoverinfo='text',
                       )
    return mods

#########################################################################################################################
# Sim track graph object traces
#########################################################################################################################
def get_simtrk_go(tree, eventidx, selection=signal_simtrk_selection, **kwargs):
    sim_pt = tree["sim_pt"].array(library="pd")[eventidx];
    sim_eta = tree["sim_eta"].array(library="pd")[eventidx];
    sim_phi = tree["sim_phi"].array(library="pd")[eventidx];
    sim_vx = tree["sim_vx"].array(library="pd")[eventidx];
    sim_vy = tree["sim_vy"].array(library="pd")[eventidx];
    sim_vz = tree["sim_vz"].array(library="pd")[eventidx];
    sim_q = tree["sim_q"].array(library="pd")[eventidx];
    sim_TC_matched = tree["sim_TC_matched"].array(library="pd")[eventidx];

    Xsim = []
    Ysim = []
    Zsim = []
    nsim = 0
    for itrk, (pt, eta, phi, vx, vy, vz, q, TC_matched) in enumerate(zip(sim_pt, sim_eta, sim_phi, sim_vx, sim_vy, sim_vz, sim_q, sim_TC_matched)):
        if not selection(pt, eta, phi, vx, vy, vz, q, TC_matched):
            continue
        # print(vx, vy, vz)
        nsim += 1
        points = m.get_helix_points(m.construct_helix_from_kinematics(pt, eta, phi, vx, vy, vz, q))
        Xsim += list(points[0]) + [None] # Non is to disconnec the lines
        Ysim += list(points[1]) + [None] # Non is to disconnec the lines
        Zsim += list(points[2]) + [None] # Non is to disconnec the lines

    sims_to_draw = go.Scatter3d(
            x = Zsim,
            y = Ysim,
            z = Xsim,
            mode='lines',
            line=dict(
                color=kwargs["color"],
                width=2,
            ),
            opacity=1,
            hoverinfo='none',
    )

    return sims_to_draw

def get_denom_simtrk_go(tree, eventidx, **kwargs):
    return get_simtrk_go(tree, eventidx, signal_simtrk_selection, **kwargs)

def get_recoed_denom_simtrk_go(tree, eventidx, **kwargs):
    return get_simtrk_go(tree, eventidx, recoed_signal_simtrk_selection, **kwargs)

#########################################################################################################################
# Line Segment object traces
#########################################################################################################################
def get_LS_go(tree, eventidx, **kwargs):
    LS_MD_idx0 = tree["LS_MD_idx0"].array(library="pd")[eventidx];
    LS_MD_idx1 = tree["LS_MD_idx1"].array(library="pd")[eventidx];
    MD_0_x = tree["MD_0_x"].array(library="pd")[eventidx];
    MD_0_y = tree["MD_0_y"].array(library="pd")[eventidx];
    MD_0_z = tree["MD_0_z"].array(library="pd")[eventidx];
    # MD_1_x = tree["MD_1_x"].array(library="pd")[eventidx];
    # MD_1_y = tree["MD_1_y"].array(library="pd")[eventidx];
    # MD_1_z = tree["MD_1_z"].array(library="pd")[eventidx];

    Xsim = []
    Ysim = []
    Zsim = []
    for iLS, (idx0, idx1) in enumerate(zip(LS_MD_idx0, LS_MD_idx1)):
        Xsim += [MD_0_x[idx0], MD_0_x[idx1]] + [None] # Non is to disconnec the lines
        Ysim += [MD_0_y[idx0], MD_0_y[idx1]] + [None] # Non is to disconnec the lines
        Zsim += [MD_0_z[idx0], MD_0_z[idx1]] + [None] # Non is to disconnec the lines

    objs_to_draw = go.Scatter3d(
            x = Zsim,
            y = Ysim,
            z = Xsim,
            mode='lines',
            line=dict(
                color=kwargs["color"],
                width=2,
            ),
            opacity=kwargs["opacity"],
            hoverinfo='none',
    )

    return objs_to_draw

def get_true_LS_go(tree, eventidx, **kwargs):
    LS_MD_idx0 = tree["LS_MD_idx0"].array(library="pd")[eventidx];
    LS_MD_idx1 = tree["LS_MD_idx1"].array(library="pd")[eventidx];
    LS_isFake = tree["LS_isFake"].array(library="pd")[eventidx];
    MD_0_x = tree["MD_0_x"].array(library="pd")[eventidx];
    MD_0_y = tree["MD_0_y"].array(library="pd")[eventidx];
    MD_0_z = tree["MD_0_z"].array(library="pd")[eventidx];
    # MD_1_x = tree["MD_1_x"].array(library="pd")[eventidx];
    # MD_1_y = tree["MD_1_y"].array(library="pd")[eventidx];
    # MD_1_z = tree["MD_1_z"].array(library="pd")[eventidx];

    Xsim = []
    Ysim = []
    Zsim = []
    for iLS, (idx0, idx1, isfake) in enumerate(zip(LS_MD_idx0, LS_MD_idx1, LS_isFake)):
        if isfake:
            continue
        Xsim += [MD_0_x[idx0], MD_0_x[idx1]] + [None] # Non is to disconnec the lines
        Ysim += [MD_0_y[idx0], MD_0_y[idx1]] + [None] # Non is to disconnec the lines
        Zsim += [MD_0_z[idx0], MD_0_z[idx1]] + [None] # Non is to disconnec the lines

    objs_to_draw = go.Scatter3d(
            x = Zsim,
            y = Ysim,
            z = Xsim,
            mode='lines',
            line=dict(
                color=kwargs["color"],
                width=2,
            ),
            opacity=kwargs["opacity"],
            hoverinfo='none',
    )

    return objs_to_draw

def get_gnn_LS_go(tree, eventidx, **kwargs):
    LS_MD_idx0 = tree["LS_MD_idx0"].array(library="pd")[eventidx];
    LS_MD_idx1 = tree["LS_MD_idx1"].array(library="pd")[eventidx];
    LS_isFake = tree["LS_isFake"].array(library="pd")[eventidx];
    MD_0_x = tree["MD_0_x"].array(library="pd")[eventidx];
    MD_0_y = tree["MD_0_y"].array(library="pd")[eventidx];
    MD_0_z = tree["MD_0_z"].array(library="pd")[eventidx];
    # MD_1_x = tree["MD_1_x"].array(library="pd")[eventidx];
    # MD_1_y = tree["MD_1_y"].array(library="pd")[eventidx];
    # MD_1_z = tree["MD_1_z"].array(library="pd")[eventidx];

    #####
    gnn_result = open("../result/csvs/data_hiddensize200_lr0.005_epoch50.csv")
    gnnres = gnn_result.readlines()[:144245]
    isTrue = []
    gnnscore = []
    for line in gnnres:
        isTrue.append(bool(int(float(line.split(",")[0]))))
        gnnscore.append(float(line.split(",")[1]))
    isTrue = np.array(isTrue)
    gnnscore = np.array(gnnscore)
    #####

    Xsim = []
    Ysim = []
    Zsim = []
    for iLS, (idx0, idx1, isfake, gnnscore) in enumerate(zip(LS_MD_idx0, LS_MD_idx1, LS_isFake)):
        if isfake:
            continue
        Xsim += [MD_0_x[idx0], MD_0_x[idx1]] + [None] # Non is to disconnec the lines
        Ysim += [MD_0_y[idx0], MD_0_y[idx1]] + [None] # Non is to disconnec the lines
        Zsim += [MD_0_z[idx0], MD_0_z[idx1]] + [None] # Non is to disconnec the lines

    objs_to_draw = go.Scatter3d(
            x = Zsim,
            y = Ysim,
            z = Xsim,
            mode='lines',
            line=dict(
                color=kwargs["color"],
                width=2,
            ),
            opacity=kwargs["opacity"],
            hoverinfo='none',
    )

    return objs_to_draw

#########################################################################################################################
# Figure layout and ranges sizes etc.
#########################################################################################################################
def config_plot_layout(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgb(0,0,0,1)",
        scene = dict(
            xaxis = dict(nticks=10, range=[-300,300],),
            yaxis = dict(nticks=10, range=[-200,200],),
            zaxis = dict(nticks=10, range=[-200,200],),
            aspectratio=dict(x=1, y=0.666, z=0.666),
        ),
        width=800,
        height=800,
        margin=dict(r=20, l=10, b=10, t=10));

def config_plot_layout_rz(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgb(0,0,0,1)",
        scene = dict(
            xaxis = dict(nticks=10, range=[-300,300],),
            yaxis = dict(nticks=10, range=[-200,200],),
            zaxis = dict(nticks=10, range=[-200,200],),
            aspectratio=dict(x=1, y=0.666, z=0.666),
        ),
        width=1400,
        height=350,
        margin=dict(r=20, l=10, b=10, t=10));

#########################################################################################################################
# Actually drawing and saving plots
#########################################################################################################################
def draw(gos, outputname):
    fig = go.Figure(gos)
    config_plot_layout(fig)
    fig.write_html(outputname)

def draw_rz(gos, outputname):
    fig = go.Figure(gos)
    config_plot_layout_rz(fig)
    fig.write_html(outputname)

def draw_signal_simtrk(tree, eventidx, outputname="signal_simtrks.html"):
    simtrks_go = get_denom_simtrk_go(tree, eventidx, color="red");
    fig = go.Figure([simtrks_go])
    config_plot_layout(fig)
    fig.write_html(outputname)

def draw_recoed_signal_simtrk(tree, eventidx, outputname="recoed_signal_simtrks.html"):
    simtrks_go = get_recoed_denom_simtrk_go(tree, eventidx, color="blue");
    fig = go.Figure([simtrks_go])
    config_plot_layout(fig)
    fig.write_html(outputname)

if __name__ == "__main__":

    f = uproot.open("/home/users/phchang/public_html/dump/forGNN/debug.root:tree")


