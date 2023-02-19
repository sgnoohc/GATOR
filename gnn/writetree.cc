#include "csv.h"
#include "TFile.h"
#include "TTree.h"
#include "TNamed.h"
#include "TString.h"
#include <iostream>
#include <vector>

int main()
{
    TString csv_path = "/blue/p.chang/p.chang/data/lst/GATOR/inference/data_hiddensize200_lr0.005_epoch50.csv";
    io::CSVReader<3> inference(csv_path.Data());
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

    TFile* file = new TFile("/blue/p.chang/p.chang/data/lst/CMSSW_12_2_0_pre2/LSTGnnNtuple_ttbar_PU200.root");
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

    // handling extra metadata information
    TNamed code_tag_data("code_tag_data", file->Get("code_tag_data")->GetTitle());
    TNamed make_log("make_log", file->Get("make_log")->GetTitle());
    TNamed gitdiff("gitdiff", file->Get("gitdiff")->GetTitle());
    TNamed input("input", file->Get("input")->GetTitle());
    TNamed full_cmd_line("full_cmd_line", file->Get("full_cmd_line")->GetTitle());
    TNamed tracklooper_path("tracklooper_path", file->Get("tracklooper_path")->GetTitle());
    TNamed gnn_csv_path("gnn_csv_path", csv_path.Data());

    code_tag_data.Write();
    make_log.Write();
    gitdiff.Write();
    input.Write();
    full_cmd_line.Write();
    tracklooper_path.Write();
    gnn_csv_path.Write();

    outtree->Write();
    ofile->Close();

}
