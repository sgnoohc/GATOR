// STL
#include <iostream>
#include <vector>
// ROOT
#include "TFile.h"
#include "TString.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
// Misc.
#include "ModuleConnectionMap.h"
#include "T3Graph.h"
#include "LST.h"
#include "json.h"
#include "tqdm.h"

class MasterModuleMap
{
private:
    std::vector<ModuleConnectionMap*> module_maps_low_pt;
    std::vector<ModuleConnectionMap*> module_maps_pos;
    std::vector<ModuleConnectionMap*> module_maps_neg;

    std::vector<ModuleConnectionMap*> load(std::vector<std::string> filenames)
    {
        std::vector<ModuleConnectionMap*> module_maps;
        for (auto filename : filenames)
        {
            module_maps.push_back(new ModuleConnectionMap(filename));
        }
        return module_maps;
    };

    std::vector<unsigned int> getConnectedModules(std::vector<ModuleConnectionMap*>& module_maps, 
                                                  float& eta, float& phi, float& dz)
    {
        // Get all connected detids
        std::vector<unsigned int> connected_detids;
        for (ModuleConnectionMap* module_map : module_maps)
        {
            std::vector<unsigned int> detids = module_map->getConnectedModules(eta, phi, dz);
            connected_detids.insert(connected_detids.end(), detids.begin(), detids.end());
        }
        return connected_detids;
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
        if (pt >= 0.9 and pt < 2.0)
        {
            return getConnectedModules(module_maps_low_pt, eta, phi, dz);
        }
        else if (pt >= 2.0)
        {
            if (charge > 0)
            {
                return getConnectedModules(module_maps_pos, eta, phi, dz);
            }
            else
            {
                return getConnectedModules(module_maps_neg, eta, phi, dz);
            }
        }
        else
        {
            return std::vector<unsigned int>();
        }
    };
};

void parseLS(LST::NTuple& lst, T3Graph::NTuple& out, 
             std::map<unsigned int, std::vector<unsigned int>>& detid_LS_map, 
             unsigned int LS_i)
{
    // Check layer of MDs in LS
    int layer0 = lst.MD_layer->at(lst.LS_MD_idx0->at(LS_i));
    int layer1 = lst.MD_layer->at(lst.LS_MD_idx1->at(LS_i));
    if (layer0 > layer1)
    {
        throw std::runtime_error("assumption that inner == idx0 and outer == idx1 was wrong!");
        return;
    }
    // Set leaves in output TTree
    out.setNodeLeaves(lst, LS_i);

    // Only populate detid map for LSs that are reasonably close to inner tracker
    // Note: barrel = 1 2 3 4 5 6, endcap = 7 8 9 10 11
    if ((layer0 > 3 && layer0 < 7) || (layer0 >= 9))
    { 
        return;
    }
    // Map detid of inner MD to LS index
    int detid = lst.MD_detid->at(lst.LS_MD_idx0->at(LS_i));
    if (detid_LS_map.find(detid) == detid_LS_map.end())
    {
        detid_LS_map[detid] = {};
    }
    detid_LS_map[detid].push_back(LS_i);
}

bool shareSimMatch(std::vector<int>& sim_idxs_1, std::vector<int>& sim_idxs_2)
{
    auto iter = std::find_first_of(
        sim_idxs_1.begin(), sim_idxs_1.end(), 
        sim_idxs_2.begin(), sim_idxs_2.end()
    );
    return (iter != sim_idxs_1.end());
}

bool shareSimMatchPixelToLS(int& pLS_nhits,
                            std::vector<int>& pLS_sim25_nhits, std::vector<int>& pLS_sim25_idxs, 
                            std::vector<int>& LS_sim50_nhits, std::vector<int>& LS_sim50_idxs)
{
    const int nhits_thresh = (pLS_nhits > 3) ? 8 : 7;
    for (unsigned int pLS_match_i = 0; pLS_match_i < pLS_sim25_idxs.size(); ++pLS_match_i)
    {
        auto iter = std::find(
            LS_sim50_idxs.begin(), LS_sim50_idxs.end(), 
            pLS_sim25_idxs[pLS_match_i]
        );
        if (iter != LS_sim50_idxs.end())
        {
            unsigned int LS_match_i = std::distance(LS_sim50_idxs.begin(), iter);
            if (pLS_sim25_nhits[pLS_match_i] + LS_sim50_nhits[LS_match_i] >= nhits_thresh)
            {
                return true;
            }
        }
    }
    return false;
}

void createT3Graph(MasterModuleMap& module_map, std::string file_name, std::string tree_name)
{
    LST::NTuple lst = LST::NTuple(TString(file_name), TString(tree_name));
    T3Graph::NTuple out = T3Graph::NTuple("output.root");

    // Loop over events
    tqdm bar;
    for (unsigned int event = 0; event < lst.n_events; ++event)
    {
        lst.init(event);
        bar.progress(event, lst.n_events);

        // Loop over T3s and find all relevant LSs
        unsigned int n_LS_in_graph = 0;
        std::map<unsigned int, unsigned int> LS_in_graph;
        std::map<unsigned int, std::vector<unsigned int>> detid_LS_map;
        for (unsigned int T3_i = 0; T3_i < lst.n_T3; ++T3_i)
        {
            // Check whether inner LS has been registered
            int inner_LS_i = lst.t3_LS_idx0->at(T3_i);
            if (LS_in_graph.find(inner_LS_i) == LS_in_graph.end())
            {
                parseLS(lst, out, detid_LS_map, inner_LS_i);
                LS_in_graph[inner_LS_i] = n_LS_in_graph;
                n_LS_in_graph++;
            }
            // Check whether outer LS has been registered
            int outer_LS_i = lst.t3_LS_idx1->at(T3_i);
            if (LS_in_graph.find(outer_LS_i) == LS_in_graph.end())
            {
                parseLS(lst, out, detid_LS_map, outer_LS_i);
                LS_in_graph[outer_LS_i] = n_LS_in_graph;
                n_LS_in_graph++;
            }
            // Set adjacency indices and edge truth label
            out.setEdgeLeaves(
                lst,                      // LST NTuple
                T3_i,                     // T3 idx in LST NTuple
                !lst.t3_isFake->at(T3_i), // is real
                LS_in_graph[inner_LS_i],  // inner node idx
                LS_in_graph[outer_LS_i]   // outer node idx
            );
        }

        // Loop over pLSs
        unsigned n_pLS_in_graph = 0;
        std::map<unsigned int, unsigned int> pLS_in_graph;
        for (unsigned int pLS_i = 0; pLS_i < lst.n_pLS; ++pLS_i)
        {
            int charge = lst.pLS_charge->at(pLS_i);
            float pt = lst.pLS_pt->at(pLS_i);
            float eta = lst.pLS_eta->at(pLS_i);
            float phi = lst.pLS_phi->at(pLS_i);
            float dz = lst.pLS_dz->at(pLS_i);
            // Loop over connected modules
            for (unsigned int& detid : module_map.getConnectedModules(charge, pt, eta, phi, dz))
            {
                // Check if any LS touch this module
                if (detid_LS_map.find(detid) == detid_LS_map.end()) { continue; }
                // Loop over LSs touching this module
                for (unsigned int& LS_i : detid_LS_map[detid])
                {
                    bool is_match = true;
                    if (is_match)
                    {
                        bool is_real = shareSimMatchPixelToLS(
                            lst.pLS_n_hits->at(pLS_i),
                            lst.pLS_all_sim25_idx->at(pLS_i), 
                            lst.pLS_all_sim25_nhits->at(pLS_i), 
                            lst.LS_all_sim50_idx->at(LS_i),
                            lst.LS_all_sim50_nhits->at(LS_i)
                        );
                        if (pLS_in_graph.find(pLS_i) == pLS_in_graph.end())
                        {
                            out.setNodeLeavesPixel(lst, pLS_i);
                            pLS_in_graph[pLS_i] = n_pLS_in_graph;
                            n_pLS_in_graph++;
                        }
                        out.setEdgeLeavesPixel(
                            lst,                                 // LST NTuple
                            pLS_i,                               // pLS idx in LST NTuple
                            is_real,                             // is real
                            n_LS_in_graph + pLS_in_graph[pLS_i], // inner node idx
                            LS_i                                 // outer node idx
                        );
                    }
                }
            }
        }

        out.fill();
        out.clear();
    }

    bar.finish();

    out.write();
    out.close();
    lst.close();
}

int main()
{
    // Initialize module maps
    MasterModuleMap module_map = MasterModuleMap();

    // Read config JSON
    std::ifstream ifs("../gnn/configs/GNN_LSnodes_T3edges.json");
    nlohmann::json config = nlohmann::json::parse(ifs);

    // Parse input files
    std::vector<std::string> file_names = config["ingress"]["input_files"];
    for (auto& file_name : file_names)
    {
        createT3Graph(module_map, file_name, config["ingress"]["ttree_name"]);
    }

    return 0;
}
