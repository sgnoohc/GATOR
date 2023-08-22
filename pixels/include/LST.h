#ifndef LST_h
#define LST_h

namespace LST
{

class NTuple
{
public:
    TFile* tfile;
    TTree* ttree;
    unsigned int n_events;
    // MD branches
    std::vector<int>* MD_layer = 0;
    std::vector<int>* MD_detid = 0;
    // LS branches
    std::vector<float>* LS_pt = 0;
    std::vector<float>* LS_eta = 0;
    std::vector<float>* LS_phi = 0;
    std::vector<float>* LS_0_x = 0;
    std::vector<float>* LS_0_y = 0;
    std::vector<float>* LS_0_z = 0;
    std::vector<float>* LS_0_layer = 0;
    std::vector<float>* LS_0_moduleType = 0;
    std::vector<float>* LS_2_x = 0;
    std::vector<float>* LS_2_y = 0;
    std::vector<float>* LS_2_z = 0;
    std::vector<float>* LS_2_layer = 0;
    std::vector<float>* LS_2_moduleType = 0;
    std::vector<int>* LS_MD_idx0 = 0;
    std::vector<int>* LS_MD_idx1 = 0;
    std::vector<std::vector<int>>* LS_all_sim50_idx = 0;
    std::vector<std::vector<int>>* LS_all_sim50_nhits = 0;
    unsigned int n_LS;
    // pLS branches
    std::vector<float>* pLS_pt = 0;
    std::vector<float>* pLS_eta = 0;
    std::vector<float>* pLS_phi = 0;
    std::vector<float>* pLS_dz = 0;
    std::vector<int>* pLS_charge = 0;
    std::vector<float>* pLS_0_x = 0;
    std::vector<float>* pLS_0_y = 0;
    std::vector<float>* pLS_0_z = 0;
    std::vector<float>* pLS_3_x = 0;
    std::vector<float>* pLS_3_y = 0;
    std::vector<float>* pLS_3_z = 0;
    std::vector<int>* pLS_n_hits = 0;
    std::vector<std::vector<int>>* pLS_all_sim25_idx = 0;
    std::vector<std::vector<int>>* pLS_all_sim25_nhits = 0;
    unsigned int n_pLS;
    // T3 branches
    std::vector<float>* t3_pt = 0;
    std::vector<float>* t3_eta = 0;
    std::vector<float>* t3_phi = 0;
    std::vector<int>* t3_LS_idx0 = 0;
    std::vector<int>* t3_LS_idx1 = 0;
    std::vector<int>* t3_isFake = 0;
    unsigned int n_T3;

    NTuple(TString file_name, TString tree_name)
    {
        // Open original ROOT file and access TTree
        tfile = new TFile(file_name);
        ttree = (TTree*)tfile->Get(tree_name);
        n_events = ttree->GetEntries();

        // MD branches
        ttree->SetBranchAddress("MD_layer", &MD_layer);
        ttree->SetBranchAddress("MD_detId", &MD_detid);
        // LS branches
        ttree->SetBranchAddress("LS_pt", &LS_pt);
        ttree->SetBranchAddress("LS_eta", &LS_eta);
        ttree->SetBranchAddress("LS_phi", &LS_phi);
        ttree->SetBranchAddress("LS_0_x", &LS_0_x);
        ttree->SetBranchAddress("LS_0_y", &LS_0_y);
        ttree->SetBranchAddress("LS_0_z", &LS_0_z);
        ttree->SetBranchAddress("LS_0_layer", &LS_0_layer);
        ttree->SetBranchAddress("LS_0_moduleType", &LS_0_moduleType);
        ttree->SetBranchAddress("LS_2_x", &LS_2_x);
        ttree->SetBranchAddress("LS_2_y", &LS_2_y);
        ttree->SetBranchAddress("LS_2_z", &LS_2_z);
        ttree->SetBranchAddress("LS_2_layer", &LS_2_layer);
        ttree->SetBranchAddress("LS_2_moduleType", &LS_2_moduleType);
        ttree->SetBranchAddress("LS_MD_idx0", &LS_MD_idx0);
        ttree->SetBranchAddress("LS_MD_idx1", &LS_MD_idx1);
        ttree->SetBranchAddress("LS_all_sim50_idx", &LS_all_sim50_idx);
        ttree->SetBranchAddress("LS_all_sim50_nhits", &LS_all_sim50_nhits);
        // pLS branches
        ttree->SetBranchAddress("pLS_pt", &pLS_pt);
        ttree->SetBranchAddress("pLS_eta", &pLS_eta);
        ttree->SetBranchAddress("pLS_phi", &pLS_phi);
        ttree->SetBranchAddress("pLS_dz", &pLS_dz);
        ttree->SetBranchAddress("pLS_charge", &pLS_charge);
        ttree->SetBranchAddress("pLS_0_x", &pLS_0_x);
        ttree->SetBranchAddress("pLS_0_y", &pLS_0_y);
        ttree->SetBranchAddress("pLS_0_z", &pLS_0_z);
        ttree->SetBranchAddress("pLS_3_x", &pLS_3_x);
        ttree->SetBranchAddress("pLS_3_y", &pLS_3_y);
        ttree->SetBranchAddress("pLS_3_z", &pLS_3_z);
        ttree->SetBranchAddress("pLS_n_hits", &pLS_n_hits);
        ttree->SetBranchAddress("pLS_all_sim25_idx", &pLS_all_sim25_idx);
        ttree->SetBranchAddress("pLS_all_sim25_nhits", &pLS_all_sim25_nhits);
        // T3 branches
        ttree->SetBranchAddress("t3_pt", &t3_pt);
        ttree->SetBranchAddress("t3_eta", &t3_eta);
        ttree->SetBranchAddress("t3_phi", &t3_phi);
        ttree->SetBranchAddress("t3_LS_idx0", &t3_LS_idx0);
        ttree->SetBranchAddress("t3_LS_idx1", &t3_LS_idx1);
        ttree->SetBranchAddress("t3_isFake", &t3_isFake);
    };

    void init(unsigned int entry)
    {
        ttree->GetEntry(entry);
        n_LS = LS_MD_idx0->size();
        n_pLS = pLS_pt->size();
        n_T3 = t3_LS_idx0->size();
    };

    void close()
    {
        tfile->Close();
    };
};

}; // End namespace LST

#endif
