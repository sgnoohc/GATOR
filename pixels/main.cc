// STL
#include <iostream>
#include <vector>
// ROOT
#include "TFile.h"
#include "TTree.h"
#include "TNamed.h"
#include "TString.h"
// Misc.
#include "ModuleConnectionMap.h"
#include "json.h"

int main()
{
    // Initialize module maps
    ModuleConnectionMap module_map_pLStoLayer1Subdet5 = ModuleConnectionMap();
    ModuleConnectionMap module_map_pLStoLayer2Subdet5 = ModuleConnectionMap();
    ModuleConnectionMap module_map_pLStoLayer1Subdet4 = ModuleConnectionMap();
    ModuleConnectionMap module_map_pLStoLayer2Subdet4 = ModuleConnectionMap();
    ModuleConnectionMap module_map_pLStoLayer1Subdet5_neg = ModuleConnectionMap();
    ModuleConnectionMap module_map_pLStoLayer2Subdet5_neg = ModuleConnectionMap();
    ModuleConnectionMap module_map_pLStoLayer1Subdet4_neg = ModuleConnectionMap();
    ModuleConnectionMap module_map_pLStoLayer2Subdet4_neg = ModuleConnectionMap();
    ModuleConnectionMap module_map_pLStoLayer1Subdet5_pos = ModuleConnectionMap();
    ModuleConnectionMap module_map_pLStoLayer2Subdet5_pos = ModuleConnectionMap();
    ModuleConnectionMap module_map_pLStoLayer1Subdet4_pos = ModuleConnectionMap();
    ModuleConnectionMap module_map_pLStoLayer2Subdet4_pos = ModuleConnectionMap();
    // Load module maps
    module_map_pLStoLayer1Subdet5.load("data/pLS_map_layer1_subdet5.txt");
    module_map_pLStoLayer2Subdet5.load("data/pLS_map_layer2_subdet5.txt");
    module_map_pLStoLayer1Subdet4.load("data/pLS_map_layer1_subdet4.txt");
    module_map_pLStoLayer2Subdet4.load("data/pLS_map_layer2_subdet4.txt");
    module_map_pLStoLayer1Subdet5_neg.load("data/pLS_map_neg_layer1_subdet5.txt");
    module_map_pLStoLayer2Subdet5_neg.load("data/pLS_map_neg_layer2_subdet5.txt");
    module_map_pLStoLayer1Subdet4_neg.load("data/pLS_map_neg_layer1_subdet4.txt");
    module_map_pLStoLayer2Subdet4_neg.load("data/pLS_map_neg_layer2_subdet4.txt");
    module_map_pLStoLayer1Subdet5_pos.load("data/pLS_map_pos_layer1_subdet5.txt");
    module_map_pLStoLayer2Subdet5_pos.load("data/pLS_map_pos_layer2_subdet5.txt");
    module_map_pLStoLayer1Subdet4_pos.load("data/pLS_map_pos_layer1_subdet4.txt");
    module_map_pLStoLayer2Subdet4_pos.load("data/pLS_map_pos_layer2_subdet4.txt");


    // Read config JSON (does this even work?)
    std::ifstream ifs("../gnn/configs/GNN_MDnodes_LSedges.json");
    nlohmann::json config = nlohmann::json::parse(ifs);


    // Read original ROOT file
    TFile* infile = new TFile("/blue/p.chang/jguiang/data/lst/GATOR/CMSSW_12_2_0_pre2/LSTNtuple_DNNTraining_hasT5Chi2_PU200.root");
    TTree* intree = (TTree*) infile->Get("tree");

    // Read MD branches of interest
    TBranch* b_MD_detid = intree->GetBranch("MD_detId");
    std::vector<int> MD_detid;
    b_MD_detid->SetAddress(&MD_detid);

    // Read LS branches of interest
    TBranch* b_LS_MD_idx0 = intree->GetBranch("LS_MD_idx0");
    std::vector<int> LS_MD_idx0;
    b_LS_MD_idx0->SetAddress(&LS_MD_idx0);
    TBranch* b_LS_MD_idx1 = intree->GetBranch("LS_MD_idx1");
    std::vector<int> LS_MD_idx1;
    b_LS_MD_idx1->SetAddress(&LS_MD_idx1);

    // Read pLS branches of interest
    TBranch* b_pLS_eta = intree->GetBranch("pLS_eta");
    std::vector<float> pLS_eta;
    b_pLS_eta->SetAddress(&pLS_eta);
    TBranch* b_pLS_phi = intree->GetBranch("pLS_phi");
    std::vector<float> pLS_phi;
    b_pLS_phi->SetAddress(&pLS_phi);
    // TBranch* b_pLS_dz = intree->GetBranch("pLS_dz");
    // std::vector<float> pLS_dz;
    // b_pLS_dz->SetAddress(&pLS_dz);
    

    // Set up output TTree
    TFile* outfile = new TFile("output.root", "recreate");
    TTree* outtree = intree->CloneTree(0);
    outtree->SetDirectory(outfile);

    // Set up LS output branches
    std::vector<float> LS_score;
    outtree->Branch("LS_score", &LS_score);

    for (int entry = 0; entry < intree->GetEntries(); ++entry)
    {
        intree->GetEntry(entry);
        int counter = 0;
        for (auto eta : pLS_eta)
        {
            std::cout << eta << std::endl;
            counter++;
            if (counter > 10) { break; }
        }
        break;
    }

    infile->Close();
    outfile->Close();

    return 0;
}
