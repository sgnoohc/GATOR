// STL
#include <iostream>
#include <vector>
// ROOT
#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
// Misc.
#include "ModuleConnectionMap.h"
#include "json.h"
#include "tqdm.h"

namespace clock = std::chrono::high_resolution_clock;

class MasterModuleMap
{
private:
    std::vector<ModuleConnectionMap> module_maps_low_pt;
    std::vector<ModuleConnectionMap> module_maps_pos;
    std::vector<ModuleConnectionMap> module_maps_neg;

    std::vector<ModuleConnectionMap> load(std::vector<std::string> filenames)
    {
        std::vector<ModuleConnectionMap> module_maps;
        for (auto filename : filenames)
        {
            module_maps.push_back(ModuleConnectionMap(filename));
        }
        return module_maps;
    };

public:
    MasterModuleMap()
    {
        module_maps_low_pt = load({
            "data/pLS_map_layer1_subdet5.txt",
            "data/pLS_map_layer2_subdet5.txt",
            "data/pLS_map_layer1_subdet4.txt",
            "data/pLS_map_layer2_subdet4.txt"
        });
        module_maps_pos = load({
            "data/pLS_map_neg_layer1_subdet5.txt",
            "data/pLS_map_neg_layer2_subdet5.txt",
            "data/pLS_map_neg_layer1_subdet4.txt",
            "data/pLS_map_neg_layer2_subdet4.txt"
        });
        module_maps_neg = load({
            "data/pLS_map_pos_layer1_subdet5.txt",
            "data/pLS_map_pos_layer2_subdet5.txt",
            "data/pLS_map_pos_layer1_subdet4.txt",
            "data/pLS_map_pos_layer2_subdet4.txt"
        });
    };

    std::vector<unsigned int> getConnectedModules(int charge, float pt, float eta, float phi, float dz)
    {
        // Get module maps based on pt, charge
        std::vector<ModuleConnectionMap> module_maps;
        if (pt >= 0.9 and pt < 2.0)
        {
            module_maps = module_maps_low_pt;
        }
        else if (pt >= 2.0)
        {
            if (charge > 0)
            {
                module_maps = module_maps_pos;
            }
            else
            {
                module_maps = module_maps_neg;
            }
        }

        // Get all connected detids
        std::vector<unsigned int> connected_detids;
        for (auto module_map : module_maps)
        {
            std::vector<unsigned int> detids = module_map.getConnectedModules(eta, phi, dz);
            connected_detids.insert(connected_detids.end(), detids.begin(), detids.end());
        }

        return connected_detids;
    };
};

int main()
{
    // Initialize module maps
    MasterModuleMap module_map = MasterModuleMap();

    // Read config JSON (does this even work?)
    std::ifstream ifs("../gnn/configs/GNN_MDnodes_LSedges.json");
    nlohmann::json config = nlohmann::json::parse(ifs);

    // Read original ROOT file
    TFile* infile = new TFile("/blue/p.chang/jguiang/data/lst/GATOR/CMSSW_12_2_0_pre2/LSTNtuple_DNNTraining_hasT5Chi2_PU200.root");
    TTreeReader reader("tree", infile);
    TTreeReaderValue<std::vector<int>> MD_detid(reader, "MD_detId");
    TTreeReaderValue<std::vector<int>> MD_layer(reader, "MD_layer");
    TTreeReaderValue<std::vector<int>> LS_MD_idx0(reader, "LS_MD_idx0");
    TTreeReaderValue<std::vector<int>> LS_MD_idx1(reader, "LS_MD_idx1");
    TTreeReaderValue<std::vector<float>> pLS_pt(reader, "pLS_pt");
    TTreeReaderValue<std::vector<float>> pLS_eta(reader, "pLS_eta");
    TTreeReaderValue<std::vector<float>> pLS_phi(reader, "pLS_phi");
    TTreeReaderValue<std::vector<float>> pLS_dz(reader, "pLS_dz");
    TTreeReaderValue<std::vector<int>> pLS_charge(reader, "pLS_charge");

    // Set up output TTree
    TFile* outfile = new TFile("output.root", "recreate");

    // Loop over events
    int n_events = reader.GetEntries();
    int event = 0;
    while (reader.Next()) 
    {
        // Organize LSs by detid of inner MD
        std::map<unsigned int, std::vector<unsigned int>> detid_LS_map;
        for (unsigned int LS_i = 0; LS_i < LS_MD_idx0->size(); ++LS_i)
        {
            // Check layer of MDs in LS
            int layer0 = MD_layer->at(LS_MD_idx0->at(LS_i));
            int layer1 = MD_layer->at(LS_MD_idx1->at(LS_i));
            if (layer0 > layer1)
            {
                throw std::runtime_error("assumption that idx0 is inner, idx1 is outer was wrong!");
            }
            // Note: barrel = 1 2 3 4 5 6, endcap = 7 8 9 10 11
            if ((layer0 >= 3 && layer0 < 7) || (layer0 >= 9))
            { 
                continue; // Skip very outer LSs
            }

            // Map detid of inner MD to LS index
            int detid = MD_detid->at(LS_MD_idx0->at(LS_i));
            if (detid_LS_map.find(detid) == detid_LS_map.end())
            {
                detid_LS_map[detid] = {};
            }
            detid_LS_map[detid].push_back(LS_i);
        }

        // Loop over pLSs
        tqdm bar;
        for (unsigned int pLS_i = 0; pLS_i < pLS_eta->size(); ++pLS_i)
        {
            bar.progress(pLS_i, pLS_eta->size());
            int charge = pLS_charge->at(pLS_i);
            float pt = pLS_pt->at(pLS_i);
            float eta = pLS_eta->at(pLS_i);
            float phi = pLS_phi->at(pLS_i);
            float dz = pLS_dz->at(pLS_i);
            // Loop over connected modules
            double step1_times = 0.;
            double step2_times = 0.;
            double step3_times = 0.;
            for (auto detid : module_map.getConnectedModules(charge, pt, eta, phi, dz))
            {
                // Check if any LS touch this module
                clock::time_point t1 = clock::now(); // DEBUG
                if (detid_LS_map.find(detid) != detid_LS_map.end())
                {
                    // Loop over LSs touching this module
                    clock::time_point t2 = clock::now(); // DEBUG
                    std::chrono::duration<double, std::milli> step1 = t2 - t1; // DEBUG
                    step1_times += step1;
                    std::vector<unsigned int> connected_LSs = detid_LS_map[detid];
                    clock::time_point t3 = clock::now(); // DEBUG
                    std::chrono::duration<double, std::milli> step2 = t3 - t2; // DEBUG
                    step2_times += step2;
                    for (auto LS_i : connected_LSs)
                    {
                        // FIXME: so some kind of consistency check between LS and pLS?
                    }
                    clock::time_point t4 = clock::now(); // DEBUG
                    std::chrono::duration<double, std::milli> step3 = t4 - t3; // DEBUG
                    step3_times += step3;
                }
            }
            bar.finish();
            std::cout << "Avg. step1 time: " << step1_times/pLS_eta->size() << std::endl;
            std::cout << "Avg. step2 time: " << step2_times/pLS_eta->size() << std::endl;
            std::cout << "Avg. step3 time: " << step3_times/pLS_eta->size() << std::endl;
        }

        event++;
    }

    infile->Close();
    outfile->Close();

    return 0;
}
