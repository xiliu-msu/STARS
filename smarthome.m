function [performance, para_rec, pp] = smarthome(all_data,pp,num_class )


    num_task = length(pp.room);
    



    % multitask + temporal consistency in loss function + linear/nonlinear independent stl model + L2(or L1) regularizer    
    data.ids_train = all_data.ids_train(pp.fold,pp.room);
    data.ids_test = all_data.ids_test(pp.fold,pp.room);
    data.X_train_common = all_data.X_train_common(pp.fold,pp.room);
    data.X_test_common = all_data.X_test_common(pp.fold,pp.room);
    data.X_train_diff = all_data.X_train_diff(pp.fold,pp.room);
    data.X_test_diff = all_data.X_test_diff(pp.fold,pp.room);
    data.Y_train = all_data.Y_train(pp.fold,pp.room);
    data.Y_test = all_data.Y_test(pp.fold,pp.room);
    data.prev_matrix_train_common = all_data.prev_matrix_train_common(pp.fold,pp.room);
    data.prev_matrix_test_common = all_data.prev_matrix_test_common(pp.fold,pp.room);
    data.prev_matrix_train_diff = all_data.prev_matrix_train_diff(pp.fold,pp.room);
    data.prev_matrix_test_diff = all_data.prev_matrix_test_diff(pp.fold,pp.room);
    data.next_matrix_train_common = all_data.next_matrix_train_common(pp.fold,pp.room);
    data.next_matrix_test_common = all_data.next_matrix_test_common(pp.fold,pp.room);
    data.next_matrix_train_diff = all_data.next_matrix_train_diff(pp.fold,pp.room);
    data.next_matrix_test_diff = all_data.next_matrix_test_diff(pp.fold,pp.room);
    data.prev_permute_train = all_data.prev_permute_train(pp.fold,pp.room);
    data.next_permute_train = all_data.next_permute_train(pp.fold,pp.room);
    all_ini = load(pp.ini_filename);
    ini.U = all_ini.ini_U(pp.fold,:);
    ini.V = all_ini.ini_V(pp.fold,:);
    ini.W_diff = all_ini.ini_W_diff(pp.fold,:);
    ini.L1 = all_ini.ini_L1(pp.fold,:);
    ini.L2 = all_ini.ini_L2(pp.fold,:);
    num_fold = size(data.ids_train,1);

    flag_w = 1;

    % training
    para_rec = cell(num_fold,1);
    for ff = 1:1:length(pp.fold)
        fprintf('fold: %d\n', ff);
        pp.ini_U = ini.U(ff,:)';
        pp.ini_V = ini.V(ff,:)';
        pp.ini_W_diff = ini.W_diff(ff,:)';
        pp.ini_L1 = ini.L1(ff,:)';
        pp.ini_L2 = ini.L2(ff,:)';
        test_see.Y_test = data.Y_test(ff,:);
        test_see.X_test_common = data.X_test_common(ff,:);
        test_see.X_test_diff = data.X_test_diff(ff,:);
        test_see.prev_matrix_test_common = data.prev_matrix_test_common(ff,:);
        test_see.prev_matrix_test_diff = data.prev_matrix_test_diff(ff,:);
        test_see.next_matrix_test_common = data.next_matrix_test_common(ff,:);
        test_see.next_matrix_test_diff = data.next_matrix_test_diff(ff,:);

        para = mtl_temporalloss_softmaxregress( test_see, data.Y_train(ff,:),data.X_train_common(ff,:),data.X_train_diff(ff,:),data.prev_matrix_train_common(ff,:),data.prev_matrix_train_diff(ff,:),data.next_matrix_train_common(ff,:),data.next_matrix_train_diff(ff,:),data.prev_permute_train(ff,:),data.next_permute_train(ff,:),flag_w,pp );

        para_rec{ff} = para;
   end

    % predicting
    est_train = cell(num_fold,num_task);
    est_test = cell(num_fold,num_task);
    for ff = 1:1:length(pp.fold)
        for t = 1:1:length(pp.room)
            est_train{ff,t} = softmax_fun([[data.X_train_common{ff,t},data.X_train_diff{ff,t}],[data.prev_matrix_train_common{ff,t},data.prev_matrix_train_diff{ff,t}]*para_rec{ff}.W{t}',[data.next_matrix_train_common{ff,t},data.next_matrix_train_diff{ff,t}]*para_rec{ff}.W{t}'],[para_rec{ff}.W{t},para_rec{ff}.L1{t},para_rec{ff}.L2{t}]);
            est_test{ff,t} = softmax_fun([[data.X_test_common{ff,t},data.X_test_diff{ff,t}],[data.prev_matrix_test_common{ff,t},data.prev_matrix_test_diff{ff,t}]*para_rec{ff}.W{t}',[data.next_matrix_test_common{ff,t},data.next_matrix_test_diff{ff,t}]*para_rec{ff}.W{t}'],[para_rec{ff}.W{t},para_rec{ff}.L1{t},para_rec{ff}.L2{t}]);
        end
    end

    % evaluate
    performance.w_brier_score_train_record = cross_validation_evaluation_weighted_brier_score( est_train,data.Y_train );
    performance.w_brier_score_test_record = cross_validation_evaluation_weighted_brier_score( est_test,data.Y_test );
    performance.brier_score_class_train_record  = cross_validation_evaluation_brier_score_class( est_train,data.Y_train );
    performance.brier_score_class_test_record  = cross_validation_evaluation_brier_score_class( est_test,data.Y_test );

   
end

