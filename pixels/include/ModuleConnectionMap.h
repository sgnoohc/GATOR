#ifndef ModuleConnectionMap_h
#define ModuleConnectionMap_h

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>

class ModuleConnectionMap
{
private:
    std::map<unsigned int, std::vector<unsigned int>> moduleConnections_;
    float pi = 3.14159265358979323846;
public:
    float n_etabins = 25.;
    float n_phibins = 72.;
    float n_dzbins = 25.;

    ModuleConnectionMap() {};
    ModuleConnectionMap(std::string filename) { load(filename); };
    ~ModuleConnectionMap() {};

    void load(std::string filename)
    {
        moduleConnections_.clear();

        std::ifstream ifile;
        ifile.open(filename.c_str());

        std::string line;
        while (std::getline(ifile, line))
        {
            unsigned int superbin;
            int number_of_connections;
            std::vector<unsigned int> connected_detids;
            unsigned int connected_detid;

            std::stringstream ss(line);

            ss >> superbin >> number_of_connections;

            for (int ii = 0; ii < number_of_connections; ++ii)
            {
                ss >> connected_detid;
                connected_detids.push_back(connected_detid);
            }
            moduleConnections_[superbin] = connected_detids;
        }
    };

    void add(std::string filename)
    {
        std::ifstream ifile;
        ifile.open(filename.c_str());
        std::string line;

        while (std::getline(ifile, line))
        {

            unsigned int superbin;
            int number_of_connections;
            std::vector<unsigned int> connected_detids;
            unsigned int connected_detid;

            std::stringstream ss(line);

            // For pixel->outer tracker, detid here is a z,eta,phi "superbin"
            ss >> superbin >> number_of_connections;

            for (int ii = 0; ii < number_of_connections; ++ii)
            {
                ss >> connected_detid;
                connected_detids.push_back(connected_detid);
            }

            // Concatenate
            moduleConnections_[superbin].insert(
                moduleConnections_[superbin].end(), 
                connected_detids.begin(), 
                connected_detids.end()
            );
            // Sort
            std::sort(
                moduleConnections_[superbin].begin(), 
                moduleConnections_[superbin].end()
            );
            // Unique
            moduleConnections_[superbin].erase(
                std::unique(
                    moduleConnections_[superbin].begin(), 
                    moduleConnections_[superbin].end()
                ), 
                moduleConnections_[superbin].end()
            );
        }
    };

    void print() 
    {
        std::cout << "Printing ModuleConnectionMap" << std::endl;
        for (auto& pair : moduleConnections_)
        {
            unsigned int superbin = pair.first;
            std::vector<unsigned int> connected_detids = pair.second;
            std::cout <<  " superbin: " << superbin <<  std::endl;
            for (auto& connected_detid : connected_detids)
            {
                std::cout <<  " connected_detid: " << connected_detid <<  std::endl;
            }
        }
    };

    const std::vector<unsigned int>& getConnectedModules(unsigned int superbin) 
    { 
        return moduleConnections_[superbin]; 
    };

    const std::vector<unsigned int>& getConnectedModules(float eta, float phi, float dz)
    {
        int etabin = (eta + 2.6) / ((2*2.6)/n_etabins);
        int phibin = (phi + pi) / ((2.*pi)/n_phibins);
        int dzbin = (dz + 30) / (2*30/n_dzbins);
        int superbin = (n_dzbins*n_phibins)*etabin + (n_dzbins)*phibin + dzbin;
        return getConnectedModules(superbin); 
    };

    const int size() { return moduleConnections_.size(); };
};

#endif
