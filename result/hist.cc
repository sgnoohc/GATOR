#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TSystem.h"
#include "TCanvas.h"

int main()
{
    TFile* f = new TFile("../gnn/output.root");
    TTree* t = (TTree*) f->Get("tree");

    TFile* o_s = new TFile("hist_sig.root", "recreate");
    TH1F* h_score_s = new TH1F("h_score", "h_score", 10000, 0., 1);
    TFile* o_b = new TFile("hist_bkg.root", "recreate");
    TH1F* h_score_b = new TH1F("h_score", "h_score", 10000, 0., 1);

    std::vector<int>* LS_isFake = 0;
    std::vector<float>* LS_score = 0;

    t->SetBranchAddress("LS_isFake", &LS_isFake);
    t->SetBranchAddress("LS_score", &LS_score);

    for (unsigned int i = 0; i < t->GetEntries(); ++i)
    {
        t->GetEntry(i);
        for (unsigned int iLS = 0; iLS < LS_isFake->size(); ++iLS)
        {
            int isFake = LS_isFake->at(iLS);
            float score = LS_score->at(iLS);
            if (isFake)
            {
                h_score_b->Fill(score);
            }
            else
            {
                h_score_s->Fill(score);
            }
        }
    }

    o_s->cd();
    h_score_s->Write();
    o_b->cd();
    h_score_b->Write();

    TCanvas* c1 = new TCanvas();
    c1->SetLogy();
    h_score_b->SetLineColor(kBlack);
    h_score_b->Draw("hist");
    h_score_s->SetLineColor(kRed);
    h_score_s->Draw("histsame");

    c1->SaveAs("hist.pdf");

}
