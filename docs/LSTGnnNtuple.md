
## Ntuple format description

LSTGnnNtuple format description

### Simulated Tracks

The following `sim_` container holds information about sub set of simulated tracks in a given event.
The selection requirement is that `sim_bunchCrossing==0` (requires that the track is in-time) and `sim_event==0` (requires that the track is from hard-scattered vertex).

| Branch Name               | Branch Type               | Description         |
| ------------------------- | ------------------------- | ------------------- |
| ```sim_pt```              | ```(vector<float>*)```          |                     |
| ```sim_eta```             | ```(vector<float>*)```          |                     |
| ```sim_phi```             | ```(vector<float>*)```          |                     |
| ```sim_pca_dxy```         | ```(vector<float>*)```          | point of closest approach dxy |
| ```sim_pca_dz```          | ```(vector<float>*)```          | point of closest approach dz  |
| ```sim_q```               | ```(vector<int>*)```            | charge              |
| ```sim_event```           | ```(vector<int>*)```            | hard-scatter if == 0 |
| ```sim_pdgId```           | ```(vector<int>*)```            | pdgid               |
| ```sim_vx```              | ```(vector<float>*)```          | production vertex x position |
| ```sim_vy```              | ```(vector<float>*)```          | production vertex x position |
| ```sim_vz```              | ```(vector<float>*)```          | production vertex x position |
| ```sim_trkNtupIdx```      | ```(vector<float>*)```          | the index in the ```sim_``` container from the actual mother tracking ntuple |
| ```sim_TC_matched```      | ```(vector<int>*)```            | == 1 if a track candidate (TC) from LST is matched to this simulated track |
| ```sim_TC_matched_mask``` | ```(vector<int>*)```            | flag for holding info on the type of TC matched (see `tc_type`) for more detailed info|

### LST's Track Candidates

| Branch Name         | Branch Type               | Description         |
| ------------------- | ------------------------- | ------------------- |
| ```tc_pt```               | ```(vector<float>*)```          | pt estimate         |
| ```tc_eta```              | ```(vector<float>*)```          | eta estimate        |
| ```tc_phi```              | ```(vector<float>*)```          | phi estimate        |
| ```tc_type```             | ```(vector<int>*)```            | type is a integer bool mask where boolean at positions, pT5 = 7, pT3 = 5, T5 = 4, pLS = 8 are set to 1 or 0 |
| ```tc_isFake```           | ```(vector<int>*)```            | true if the track candidate is not matched a true ```sim_``` |
| ```tc_isDuplicate```      | ```(vector<int>*)```            | ture if the track candidate is true, but another tc is also matched to the same ```sim_```
| ```tc_matched_simIdx```   | ```(vector<vector<int> >*)```   | the indices to the ```sim_``` container that this tc is matched to. N.B. However, many of the matched true ```sim_``` information is lost as we do not save all of the simulated tracks to ```sim_``` container |

| Branch Name         | Branch Type               | Description         |
| ------------------- | ------------------------- | ------------------- |
| ```MD_pt```               | ```(vector<float>*)```          | pt estimate         |
| ```MD_eta```              | ```(vector<float>*)```          | eta estimate        |
| ```MD_phi```              | ```(vector<float>*)```          | phi estimate        |
| ```MD_dphichange```       | ```(vector<float>*)```          | dphichange          |
| ```MD_isFake```           | ```(vector<int>*)```            | true if both hits are not matched to same ```sim_``` |
| ```MD_tpType```           | ```(vector<int>*)```            | see getDenomSimTrkType from ```code/core/trkCore.cc``` |
| ```MD_detId```            | ```(vector<int>*)```            | detId of where the mini-doublet is from |
| ```MD_layer```            | ```(vector<int>*)```            | layer of where the mini-doublet is from (1 2 3 4 5 6 for barrel, 7 8 9 10 11 for endcap) |
| ```MD_0_r```              | ```(vector<float>*)```          | lower hit's radius  |
| ```MD_0_x```              | ```(vector<float>*)```          | lower hit's x       |
| ```MD_0_y```              | ```(vector<float>*)```          | lower hit's y       |
| ```MD_0_z```              | ```(vector<float>*)```          | lower hit's z       |
| ```MD_1_r```              | ```(vector<float>*)```          | upper hit's radius  |
| ```MD_1_x```              | ```(vector<float>*)```          | upper hit's x       |
| ```MD_1_y```              | ```(vector<float>*)```          | upper hit's y       |
| ```MD_1_z```              | ```(vector<float>*)```          | upper hit's z       |

| Branch Name         | Branch Type               | Description         |
| ------------------- | ------------------------- | ------------------- |
| ```LS_pt```               | ```(vector<float>*)```          | pt estimate         |
| ```LS_eta```              | ```(vector<float>*)```          | eta estimate        |
| ```LS_phi```              | ```(vector<float>*)```          | phi estimate        |
| ```LS_isFake```           | ```(vector<int>*)```            | true if all four hits are not matched to same ```sim_``` |
| ```LS_MD_idx0```          | ```(vector<int>*)```            | index to the lower MD object in ```MD_``` container |
| ```LS_MD_idx1```          | ```(vector<int>*)```            | index to the upper MD object in ```MD_``` container |
| ```LS_sim_pt```           | ```(vector<float>*)```          | pt of the first matched sim-trk |
| ```LS_sim_eta```          | ```(vector<float>*)```          | eta of the first matched sim-trk |
| ```LS_sim_phi```          | ```(vector<float>*)```          | phi of the first matched sim-trk |
| ```LS_sim_pca_dxy```      | ```(vector<float>*)```          | pca_dxy of the first matched sim-trk |
| ```LS_sim_pca_dz```       | ```(vector<float>*)```          | pca_dz of the first matched sim-trk |
| ```LS_sim_q```            | ```(vector<int>*)```            | charge of the first matched sim-trk |
| ```LS_sim_pdgId```        | ```(vector<int>*)```            | pdgId of the first matched sim-trk |
| ```LS_sim_event```        | ```(vector<int>*)```            | == 0 if the first matched sim-trk is hard scatter |
| ```LS_sim_bx```           | ```(vector<int>*)```            | == 0 if the first matched sim-trk is in-time |
| ```LS_sim_vx```           | ```(vector<float>*)```          | x of the production vertex of the first matched sim-trk |
| ```LS_sim_vy```           | ```(vector<float>*)```          | y of the production vertex of the first matched sim-trk |
| ```LS_sim_vz```           | ```(vector<float>*)```          | z of the production vertex of the first matched sim-trk |
| ```LS_isInTrueTC```       | ```(vector<int>*)```            | if the LS is part of a true matched TC in the ```tc_``` container |
