#include "csv.h"
#include "TFile.h"
#include "TTree.h"
#include <iostream>
#include <vector>

int main()
{
    io::CSVReader<3> inference("/home/p.chang/data/lst/GATOR/inference/data_hiddensize200_lr0.005_epoch50.csv");
    std::vector<std::vector<int>> eventid_vec_vec;
    std::vector<std::vector<float>> istrue_vec_vec;
    std::vector<std::vector<float>> score_vec_vec;
    std::vector<int> eventid_vec;
    std::vector<float> istrue_vec;
    std::vector<float> score_vec;

    int current_event_index = 0;
    int eventid;
    float istrue;
    float score;
    while (inference.read_row(eventid, istrue, score))
    {
        if (current_event_index != eventid)
        {
            eventid_vec_vec.push_back(eventid_vec);
            istrue_vec_vec.push_back(istrue_vec);
            score_vec_vec.push_back(score_vec);
            eventid_vec.clear();
            istrue_vec.clear();
            score_vec.clear();
            current_event_index++;
        }
        eventid_vec.push_back(eventid);
        istrue_vec.push_back(istrue);
        score_vec.push_back(score);
    }
    eventid_vec_vec.push_back(eventid_vec);
    istrue_vec_vec.push_back(istrue_vec);
    score_vec_vec.push_back(score_vec);
    eventid_vec.clear();
    istrue_vec.clear();
    score_vec.clear();

    TFile* file = new TFile("/home/p.chang/data/lst/CMSSW_12_2_0_pre2/LSTGnnNtuple_ttbar_PU200.root");
    TTree* tree = (TTree*) file->Get("tree");

    TFile* ofile = new TFile("output.root", "recreate");
    TTree* outtree = tree->CloneTree(0);
    outtree->SetDirectory(ofile);

    std::vector<float>* LS_score = 0;
    outtree->Branch("LS_score", &LS_score);

    int begin_event_idx = 95;
    int end_event_idx = 100;

    int this_event = 0;

    for (int event_idx = begin_event_idx; event_idx < end_event_idx; ++event_idx)
    {
        tree->GetEntry(event_idx);
        LS_score = &score_vec_vec[this_event];
        this_event++;
        outtree->Fill();
    }

    outtree->Write();
    ofile->Close();

}
