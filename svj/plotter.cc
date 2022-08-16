void plotter() {
    // vector<TString> bkgnum = {"364703", "364704", "364705", "364706", "364707", "364708", "364709", "364710", "364711", "364712"};
    vector<TString> bkgnum = {"364703", "364704", "364705", "364706", "364707", "364708", "364709"};
    // vector<double>  pseudo_scale = {1.0, 2.0, 6.0, 20.0, 60.0, 400.0, 2800.0, 19600.0, 137200.0, 960400.0};

    vector<double> xsec = {(2.6450e+04, 2.5461e+02, 4.5532e+00, 2.5754e-01, 1.6215e-02, 6.2506e-04, 1.9639e-05)*1e3}; // pb JZ03-9
    vector<double> eff  = {(1.165838e-02, 1.336560e-02, 1.452648e-02, 9.471878e-03, 1.1097e-02, 1.015436e-02, 1.2056e-02)}; // JZ03-9
    vector<double> meta_num_events = {614120800, 441616600, 177251284, 31961550, 15979000, 15963000, 7631000};
    vector<double> meta_sum_weights = {4473.89, 98.4767, 2.9619, 0.0889688, 0.0122396, 0.00354308, 0.000656467};
    const auto lumi_a = 36.2*1e6; // pb^-1

    // TFile* w_out = new TFile("weighted_variables.root","recreate");
    // TCanvas *c1 = new TCanvas("c", "canvas", 800, 800);
    // std::vector<double> weighted_pt;
    // TH1D* pt_plot = new TH1D("hhh","pT",1000,0.0,3000);

    TH1D* pt_plot = new TH1D("pt","Leading Jet pT",100.0,0.0,6000.0);
    pt_plot->Sumw2();
    // TH1D* pt_plot = new TH1D("pt","Leading Jet pT",200.0,0.0,6000.0);
    // pt_plot->Sumw2();
    // TH1D* pt_plot = new TH1D("pt","Leading Jet pT",200.0,0.0,6000.0);
    // pt_plot->Sumw2();
    // TH1D* pt_plot = new TH1D("pt","Leading Jet pT",200.0,0.0,6000.0);
    // pt_plot->Sumw2();
    // TH1D* pt_plot = new TH1D("pt","Leading Jet pT",200.0,0.0,6000.0);
    // pt_plot->Sumw2();

    for (int g = 0; g < bkgnum.size(); ++g){
        TFile *bkg = TFile::Open("~/Downloads/svj_data/user.ebusch." + bkgnum[g] + ".root");
        TTree *t1 = (TTree*) bkg->Get("PostSel");
        TH1F  *cutflow_hist = (TH1F*) bkg->Get("cutflow");
        Double_t all_num_events = cutflow_hist->GetBinContent(1);
        // Double_t cuts_num_events = cutflow_hist->GetBinContent(3);
        std::cout << "orig_num_events: " << cutflow_hist->GetBinContent(1) << std::endl;
        Double_t weight=0;
        Double_t pt=0;
        // Double_t weights_sum=0;
        t1->SetBranchAddress("mcEventWeight",&weight);
        t1->SetBranchAddress("jet1_pt",&pt);

        // for(int j=0; j < t1->GetEntries(); j++){
        //     t1->GetEntry(j);
        //     weights_sum+=weight;
        // }
        // TH1D* pt_plot = new TH1D("pt","Leading Jet pT",600.0,0.0,5000.0);

        for(int i=0; i < t1->GetEntries(); i++){
            t1->GetEntry(i);
            // pt_plot->Fill(pt/weights_sum, weight);
            // pt_plot->Fill(pt, weight/(binContent));
            // pt_plot->Fill(pt, (weight*xsec[g]*eff[g])/sum_weights[g]);
            pt_plot->Fill(pt, (weight*xsec[g]*eff[g])/((all_num_events/meta_num_events[g])*meta_sum_weights[g]));
            // pt_plot->Fill(pt, weight/(binContent*pseudo_scale[g]));

        }
        // c1->cd();
        // pt_plot->Draw("hist e same");
    }

    pt_plot->Draw("E");
    // w_out->WriteTObject(pt_plot);


   


}
