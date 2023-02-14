
## Ntuple format description

LSTGnnNtuple format description

### Simulated Tracks


The following `sim_` container holds information about sub set of simulated tracks in a given event.
The selection requirement is that `sim_bunchCrossing==0` (requires that the track is in-time) and `sim_event==0` (requires that the track is from hard-scattered vertex).

    | Branch Name         | Branch Type               | Description         |
    | ------------------- | ------------------------- | ------------------- |
    | sim_pt              | (vector<float>*)          |                     |
    | sim_eta             | (vector<float>*)          |                     |
    | sim_phi             | (vector<float>*)          |                     |
    | sim_pca_dxy         | (vector<float>*)          | point of closest approach dxy |
    | sim_pca_dz          | (vector<float>*)          | point of closest approach dz  |
    | sim_q               | (vector<int>*)            | charge              |
    | sim_event           | (vector<int>*)            | hard-scatter if == 0 |
    | sim_pdgId           | (vector<int>*)            | pdgid               |
    | sim_vx              | (vector<float>*)          | production vertex x position |
    | sim_vy              | (vector<float>*)          | production vertex x position |
    | sim_vz              | (vector<float>*)          | production vertex x position |
    | sim_trkNtupIdx      | (vector<float>*)          | the index in the `sim_` container from the actual mother tracking ntuple |
    | sim_TC_matched      | (vector<int>*)            | == 1 if a track candidate (TC) from LST is matched to this simulated track |
    | sim_TC_matched_mask | (vector<int>*)            | flag for holding info on the type of TC matched |

    | Branch Name         | Branch Type               | Description         |
    | ------------------- | ------------------------- | ------------------- |
    | tc_pt               | (vector<float>*)          | tc_pt               |
    | tc_eta              | (vector<float>*)          | tc_eta              |
    | tc_phi              | (vector<float>*)          | tc_phi              |
    | tc_type             | (vector<int>*)            | tc_type             |
    | tc_isFake           | (vector<int>*)            | tc_isFake           |
    | tc_isDuplicate      | (vector<int>*)            | tc_isDuplicate      |
    | tc_matched_simIdx   | (vector<vector<int> >*)   | tc_matched_simIdx   |

    | Branch Name         | Branch Type               | Description         |
    | ------------------- | ------------------------- | ------------------- |
    | MD_pt               | (vector<float>*)          | MD_pt               |
    | MD_eta              | (vector<float>*)          | MD_eta              |
    | MD_phi              | (vector<float>*)          | MD_phi              |
    | MD_dphichange       | (vector<float>*)          | MD_dphichange       |
    | MD_isFake           | (vector<int>*)            | MD_isFake           |
    | MD_tpType           | (vector<int>*)            | MD_tpType           |
    | MD_detId            | (vector<int>*)            | MD_detId            |
    | MD_layer            | (vector<int>*)            | MD_layer            |
    | MD_0_r              | (vector<float>*)          | MD_0_r              |
    | MD_0_x              | (vector<float>*)          | MD_0_x              |
    | MD_0_y              | (vector<float>*)          | MD_0_y              |
    | MD_0_z              | (vector<float>*)          | MD_0_z              |
    | MD_1_r              | (vector<float>*)          | MD_1_r              |
    | MD_1_x              | (vector<float>*)          | MD_1_x              |
    | MD_1_y              | (vector<float>*)          | MD_1_y              |
    | MD_1_z              | (vector<float>*)          | MD_1_z              |

    | Branch Name         | Branch Type               | Description         |
    | ------------------- | ------------------------- | ------------------- |
    | LS_pt               | (vector<float>*)          | LS_pt               |
    | LS_eta              | (vector<float>*)          | LS_eta              |
    | LS_phi              | (vector<float>*)          | LS_phi              |
    | LS_isFake           | (vector<int>*)            | LS_isFake           |
    | LS_MD_idx0          | (vector<int>*)            | LS_MD_idx0          |
    | LS_MD_idx1          | (vector<int>*)            | LS_MD_idx1          |
    | LS_sim_pt           | (vector<float>*)          | LS_sim_pt           |
    | LS_sim_eta          | (vector<float>*)          | LS_sim_eta          |
    | LS_sim_phi          | (vector<float>*)          | LS_sim_phi          |
    | LS_sim_pca_dxy      | (vector<float>*)          | LS_sim_pca_dxy      |
    | LS_sim_pca_dz       | (vector<float>*)          | LS_sim_pca_dz       |
    | LS_sim_q            | (vector<int>*)            | LS_sim_q            |
    | LS_sim_pdgId        | (vector<int>*)            | LS_sim_pdgId        |
    | LS_sim_event        | (vector<int>*)            | LS_sim_event        |
    | LS_sim_bx           | (vector<int>*)            | LS_sim_bx           |
    | LS_sim_vx           | (vector<float>*)          | LS_sim_vx           |
    | LS_sim_vy           | (vector<float>*)          | LS_sim_vy           |
    | LS_sim_vz           | (vector<float>*)          | LS_sim_vz           |
    | LS_isInTrueTC       | (vector<int>*)            | LS_isInTrueTC       |
