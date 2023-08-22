#ifndef T3GRAPH_H
#define T3GRAPH_H

#include "LST.h"

namespace T3Graph
{

struct NTuple
{
    TFile* tfile;
    TTree* ttree;
    // Edge features
    std::vector<float> t3_pt;
    std::vector<float> t3_eta;
    std::vector<float> t3_phi;
    std::vector<int> t3_hasPixel;
    std::vector<int> t3_isFake;
    // Adjacency indices
    std::vector<int> t3_xLS_idx0;
    std::vector<int> t3_xLS_idx1;
    // Node features
    std::vector<int> xLS_isPixel;
    std::vector<float> xLS_pt;
    std::vector<float> xLS_eta;
    std::vector<float> xLS_phi;
    std::vector<float> xLS_innerHit_x;
    std::vector<float> xLS_innerHit_y;
    std::vector<float> xLS_innerHit_z;
    std::vector<int> xLS_innerHit_layer;
    std::vector<int> xLS_innerHit_moduleType;
    std::vector<float> xLS_outerHit_x;
    std::vector<float> xLS_outerHit_y;
    std::vector<float> xLS_outerHit_z;
    std::vector<int> xLS_outerHit_layer;
    std::vector<int> xLS_outerHit_moduleType;
    // Other branches
    int n_LS = 0;
    int n_pLS = 0;
    int n_xLS = 0;
    int n_edge_total = 0;
    int n_fake_edge_total = 0;
    int n_real_edge_total = 0;
    int n_edge_noPixel = 0;
    int n_fake_edge_noPixel = 0;
    int n_real_edge_noPixel = 0;
    int n_edge_pixel = 0;
    int n_fake_edge_pixel = 0;
    int n_real_edge_pixel = 0;

    NTuple(TString file_name, TString tree_name = "tree")
    {
        tfile = new TFile(file_name, "RECREATE");
        ttree = new TTree(tree_name, tree_name);
        // Edge features
        ttree->Branch("t3_pt", &t3_pt);
        ttree->Branch("t3_eta", &t3_eta);
        ttree->Branch("t3_phi", &t3_phi);
        ttree->Branch("t3_hasPixel", &t3_hasPixel);
        ttree->Branch("t3_isFake", &t3_isFake);
        // Adjacency indices
        ttree->Branch("t3_xLS_idx0", &t3_xLS_idx0);
        ttree->Branch("t3_xLS_idx1", &t3_xLS_idx1);
        // Node features
        ttree->Branch("xLS_isPixel", &xLS_isPixel);
        ttree->Branch("xLS_pt", &xLS_pt);
        ttree->Branch("xLS_eta", &xLS_eta);
        ttree->Branch("xLS_phi", &xLS_phi);
        ttree->Branch("xLS_innerHit_x", &xLS_innerHit_x);
        ttree->Branch("xLS_innerHit_y", &xLS_innerHit_y);
        ttree->Branch("xLS_innerHit_z", &xLS_innerHit_z);
        ttree->Branch("xLS_innerHit_layer", &xLS_innerHit_layer);
        ttree->Branch("xLS_innerHit_moduleType", &xLS_innerHit_moduleType);
        ttree->Branch("xLS_outerHit_x", &xLS_outerHit_x);
        ttree->Branch("xLS_outerHit_y", &xLS_outerHit_y);
        ttree->Branch("xLS_outerHit_z", &xLS_outerHit_z);
        ttree->Branch("xLS_outerHit_layer", &xLS_outerHit_layer);
        ttree->Branch("xLS_outerHit_moduleType", &xLS_outerHit_moduleType);
        // Other branches
        ttree->Branch("n_LS", &n_LS);
        ttree->Branch("n_pLS", &n_pLS);
        ttree->Branch("n_xLS", &n_xLS);
        ttree->Branch("n_edge_total", &n_edge_total);
        ttree->Branch("n_fake_edge_total", &n_fake_edge_total);
        ttree->Branch("n_real_edge_total", &n_real_edge_total);
        ttree->Branch("n_edge_pixel", &n_edge_pixel);
        ttree->Branch("n_fake_edge_pixel", &n_fake_edge_pixel);
        ttree->Branch("n_real_edge_pixel", &n_real_edge_pixel);
        ttree->Branch("n_edge_noPixel", &n_edge_noPixel);
        ttree->Branch("n_fake_edge_noPixel", &n_fake_edge_noPixel);
        ttree->Branch("n_real_edge_noPixel", &n_real_edge_noPixel);
    };

    void clear()
    {
        // Edge features
        t3_pt = { -999 };
        t3_eta = { -999 };
        t3_phi = { -999 };
        t3_hasPixel = { -999 };
        t3_isFake = { -999 };
        // Adjacency indices
        t3_xLS_idx0 = { -999 };
        t3_xLS_idx1 = { -999 };
        // Node features
        xLS_isPixel = { -999 };
        xLS_pt = { -999 };
        xLS_eta = { -999 };
        xLS_phi = { -999 };
        xLS_innerHit_x = { -999 };
        xLS_innerHit_y = { -999 };
        xLS_innerHit_z = { -999 };
        xLS_innerHit_layer = { -999 };
        xLS_innerHit_moduleType = { -999 };
        xLS_outerHit_x = { -999 };
        xLS_outerHit_y = { -999 };
        xLS_outerHit_z = { -999 };
        xLS_outerHit_layer = { -999 };
        xLS_outerHit_moduleType = { -999 };
        // Other branches
        n_LS = 0;
        n_pLS = 0;
        n_xLS = 0;
        n_edge_total = 0;
        n_fake_edge_total = 0;
        n_real_edge_total = 0;
        n_edge_noPixel = 0;
        n_fake_edge_noPixel = 0;
        n_real_edge_noPixel = 0;
        n_edge_pixel = 0;
        n_fake_edge_pixel = 0;
        n_real_edge_pixel = 0;
    };

    void setEdgeLeaves(LST::NTuple& lst, unsigned int T3_i, bool is_real,
                       unsigned int inner_idx, unsigned int outer_idx)
    {
        t3_pt.push_back(lst.t3_pt->at(T3_i));
        t3_eta.push_back(lst.t3_eta->at(T3_i));
        t3_phi.push_back(lst.t3_phi->at(T3_i));
        t3_xLS_idx0.push_back(inner_idx);
        t3_xLS_idx1.push_back(outer_idx);
        t3_hasPixel.push_back(false);
        t3_isFake.push_back(!is_real);

        n_edge_total++;
        n_fake_edge_total += !is_real;
        n_real_edge_total += is_real;
        n_edge_noPixel++;
        n_fake_edge_noPixel += !is_real;
        n_real_edge_noPixel += is_real;
    };

    void setNodeLeaves(LST::NTuple& lst, unsigned int LS_i)
    {
        // Set xLS leaves
        xLS_pt.push_back(lst.LS_pt->at(LS_i));
        xLS_eta.push_back(lst.LS_eta->at(LS_i));
        xLS_phi.push_back(lst.LS_phi->at(LS_i));
        xLS_isPixel.push_back(false);
        xLS_innerHit_x.push_back(lst.LS_0_x->at(LS_i));
        xLS_innerHit_y.push_back(lst.LS_0_y->at(LS_i));
        xLS_innerHit_z.push_back(lst.LS_0_z->at(LS_i));
        xLS_innerHit_layer.push_back(lst.LS_0_layer->at(LS_i));
        xLS_innerHit_moduleType.push_back(lst.LS_0_moduleType->at(LS_i));
        xLS_outerHit_x.push_back(lst.LS_2_x->at(LS_i));
        xLS_outerHit_y.push_back(lst.LS_2_y->at(LS_i));
        xLS_outerHit_z.push_back(lst.LS_2_z->at(LS_i));
        xLS_outerHit_layer.push_back(lst.LS_2_layer->at(LS_i));
        xLS_outerHit_moduleType.push_back(lst.LS_2_moduleType->at(LS_i));

        n_LS++;
        n_xLS++;
    };

    void setEdgeLeavesPixel(LST::NTuple& lst, unsigned int pLS_i, bool is_real, 
                            unsigned int inner_idx, unsigned int outer_idx)
    {
        t3_pt.push_back(lst.pLS_pt->at(pLS_i));
        t3_eta.push_back(lst.pLS_eta->at(pLS_i));
        t3_phi.push_back(lst.pLS_phi->at(pLS_i));
        t3_xLS_idx0.push_back(inner_idx);
        t3_xLS_idx1.push_back(outer_idx);
        t3_hasPixel.push_back(true);
        t3_isFake.push_back(!is_real);

        n_edge_total++;
        n_fake_edge_total += !is_real;
        n_real_edge_total += is_real;
        n_edge_pixel++;
        n_fake_edge_pixel += !is_real;
        n_real_edge_pixel += is_real;
    };

    void setNodeLeavesPixel(LST::NTuple& lst, unsigned int pLS_i)
    {
        // Set xLS leaves
        xLS_pt.push_back(lst.pLS_pt->at(pLS_i));
        xLS_eta.push_back(lst.pLS_eta->at(pLS_i));
        xLS_phi.push_back(lst.pLS_phi->at(pLS_i));
        xLS_isPixel.push_back(true);
        xLS_innerHit_x.push_back(lst.pLS_0_x->at(pLS_i));
        xLS_innerHit_y.push_back(lst.pLS_0_y->at(pLS_i));
        xLS_innerHit_z.push_back(lst.pLS_0_z->at(pLS_i));
        xLS_innerHit_layer.push_back(0);
        xLS_innerHit_moduleType.push_back(0);
        xLS_outerHit_x.push_back(lst.pLS_3_x->at(pLS_i));
        xLS_outerHit_y.push_back(lst.pLS_3_y->at(pLS_i));
        xLS_outerHit_z.push_back(lst.pLS_3_z->at(pLS_i));
        xLS_outerHit_layer.push_back(0);
        xLS_outerHit_moduleType.push_back(0);

        n_pLS++;
        n_xLS++;
    };

    void fill()
    {
        ttree->Fill();
    };

    void write()
    {
        ttree->Write();
    };

    void close()
    {
        tfile->Close();
    };
};

}; // End namespace T3Graph

#endif
