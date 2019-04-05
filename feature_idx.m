function  idx  = feature_idx(const)
    idx.ind_location = 180:248;
    ind_acc = 109:179;
    ind_cam_hallway = [1:2,7:9,16:17,22:24,31:32,37:39,46:48,55:58,67:69,76:79,88:90,97:100,379:443];
    ind_cam_kitchen = [3:4,10:12,18:19,25:27,33:34,40:42,49:51,59:62,70:72,80:83,91:93,101:104,314:378];
    ind_cam_living = [5:6,13:15,20:21,28:30,35:36,43:45,52:54,63:66,73:75,84:87,94:96,105:108,249:313];
    ind_constant = 444; 
    
    idx.ind_diff = cell(9,1);
    idx.ind_diff{1} = [];
    idx.ind_diff{2} = [];
    idx.ind_diff{3} = [];
    idx.ind_diff{4} = ind_cam_hallway;
    idx.ind_diff{5} = ind_cam_kitchen;
    idx.ind_diff{6} = ind_cam_living;
    idx.ind_diff{7} = [];
    idx.ind_diff{8} = [];
    idx.ind_diff{9} = [];
    idx.ind_common = [ind_acc,idx.ind_location,ind_constant];


end
