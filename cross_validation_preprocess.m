function  output = cross_validation_preprocess( method,filename,idx,num_task )
    data = load(filename);


    % load the data
    X_train_folders = data.X_train_folders;
    X_test_folders = data.X_test_folders;
    prob_train_folders = data.prob_train_folders;
    prob_test_folders = data.prob_test_folders;
    loc_prob_train_folders = data.loc_prob_train_folders;
    loc_prob_test_folders = data.loc_prob_test_folders;
    ids_train_folders = data.ids_train_folders;
    ids_test_folders = data.ids_test_folders;
    num_fold = length(X_train_folders);
    
    
    % initilize the output
    output.ids_train = cell(num_fold,num_task);
    output.ids_test = cell(num_fold,num_task);
    if (ismember(method,[0,3]))
        output.X_train = cell(num_fold,num_task);
        output.X_test = cell(num_fold,num_task);       
        output.Y_train = cell(num_fold,num_task);
        output.Y_test = cell(num_fold,num_task);
    elseif (ismember(method,[1,4]))
        output.X_train = cell(num_fold,num_task);
        output.X_test = cell(num_fold,num_task);
        output.prev_matrix_train = cell(num_fold,num_task);
        output.prev_matrix_test = cell(num_fold,num_task);
        output.next_matrix_train = cell(num_fold,num_task);
        output.next_matrix_test = cell(num_fold,num_task);
        output.Y_train = cell(num_fold,num_task);
        output.Y_test = cell(num_fold,num_task);
        output.prev_permute_train = cell(num_fold,num_task);
        output.next_permute_train = cell(num_fold,num_task);
    elseif (ismember(method,[2,5,6,7,8]))
        output.X_train_common = cell(num_fold,num_task);
        output.X_test_common = cell(num_fold,num_task);
        output.X_train_diff = cell(num_fold,num_task);
        output.X_test_diff = cell(num_fold,num_task);
        output.prev_matrix_train_common = cell(num_fold,num_task);
        output.prev_matrix_test_common = cell(num_fold,num_task);
        output.prev_matrix_train_diff = cell(num_fold,num_task);
        output.prev_matrix_test_diff = cell(num_fold,num_task);
        output.next_matrix_train_common = cell(num_fold,num_task);
        output.next_matrix_test_common = cell(num_fold,num_task);
        output.next_matrix_train_diff = cell(num_fold,num_task);
        output.next_matrix_test_diff = cell(num_fold,num_task);
        output.Y_train = cell(num_fold,num_task);
        output.Y_test = cell(num_fold,num_task);
        output.prev_permute_train = cell(num_fold,num_task);
        output.next_permute_train = cell(num_fold,num_task);
    end
            
    
    
    
    % for each fold
    for ff = 1:1:num_fold
        % prepare the matrices for preprocess.m
        if (ismember(method,[0,3]))
            matrix_train_ff = {X_train_folders{ff}};
            matrix_test_ff = {X_test_folders{ff}};
            target_train_ff = {prob_train_folders{ff}};
            target_test_ff = {prob_test_folders{ff}};
        elseif (ismember(method,[1,2,4,5]))
            [prev_permute_ff, next_permute_ff,~,~] = seq_relation([ids_train_folders{ff};ids_test_folders{ff}]);
            prev_matrix_ff = prev_permute_ff*[X_train_folders{ff};X_test_folders{ff}];
            next_matrix_ff = next_permute_ff*[X_train_folders{ff};X_test_folders{ff}];
            prev_matrix_train_ff = prev_matrix_ff(1:size(X_train_folders{ff},1),:);
            next_matrix_train_ff = next_matrix_ff(1:size(X_train_folders{ff},1),:);
            prev_matrix_test_ff = prev_matrix_ff(size(X_train_folders{ff},1)+1:end,:);
            next_matrix_test_ff = next_matrix_ff(size(X_train_folders{ff},1)+1:end,:);
            matrix_train_ff = {X_train_folders{ff};prev_matrix_train_ff;next_matrix_train_ff};
            matrix_test_ff = {X_test_folders{ff};prev_matrix_test_ff;next_matrix_test_ff};
            target_train_ff = {prob_train_folders{ff}};
            target_test_ff = {prob_test_folders{ff}};
        end

        % call preprocess.m
        [ matrix_train_tasks_ff, matrix_test_tasks_ff, Y_train_tasks_ff, Y_test_tasks_ff, ~, ~, ids_train_tasks_ff, ids_test_tasks_ff] = preprocess( matrix_train_ff, matrix_test_ff, ...
            target_train_ff, target_test_ff, ...
            loc_prob_train_folders{ff}, loc_prob_test_folders{ff},...
            ids_train_folders{ff},ids_test_folders{ff},...
            num_task, idx.ind_location, 1, 'mean' );
        
        
        
        % reorganize the matrices
        if (ismember(method,[0,3]))
            X_train_tasks_ff = matrix_train_tasks_ff(:,1);
            X_test_tasks_ff = matrix_test_tasks_ff(:,1);
        elseif (ismember(method,[1,2,4,5,6,7,8]))
            X_train_tasks_ff = matrix_train_tasks_ff(:,1);
            prev_matrix_train_tasks_ff = matrix_train_tasks_ff(:,2);
            next_matrix_train_tasks_ff = matrix_train_tasks_ff(:,3);
            X_test_tasks_ff = matrix_test_tasks_ff(:,1);
            prev_matrix_test_tasks_ff = matrix_test_tasks_ff(:,2);
            next_matrix_test_tasks_ff = matrix_test_tasks_ff(:,3);
        end
                    
        
        % for each task
        for t = 1:1:num_task
            % find the ids of prevousious seconds, and next seconds in
            % train or test
            [~,~,P_avlbl_ids_train,N_avlbl_ids_train] = seq_relation(ids_train_tasks_ff{t});   % find the seconds whose both previous second and next second can be found in training data
            [~,~,P_avlbl_ids_test,N_avlbl_ids_test] = seq_relation(ids_test_tasks_ff{t});
            % remove those ids which has their previous second and next
            % second NOT in train or test
            eff_train_rows = find(ismember(ids_train_tasks_ff{t},intersect(P_avlbl_ids_train,N_avlbl_ids_train,'rows'),'rows'));
            eff_test_rows = find(ismember(ids_test_tasks_ff{t},intersect(P_avlbl_ids_test,N_avlbl_ids_test,'rows'),'rows'));
            

            % get the output
            output.ids_train{ff,t} = ids_train_tasks_ff{t}(eff_train_rows,:);
            output.ids_test{ff,t} = ids_test_tasks_ff{t}(eff_test_rows,:);
            if (ismember(method,[0,3]))
                output.X_train{ff,t} = X_train_tasks_ff{t}(eff_train_rows,[idx.ind_common,idx.ind_diff{t}]);
                output.X_test{ff,t} = X_test_tasks_ff{t}(eff_test_rows,[idx.ind_common,idx.ind_diff{t}]);         % use only available features
                output.Y_train{ff,t} = Y_train_tasks_ff{t}(eff_train_rows,:);
                output.Y_test{ff,t} = Y_test_tasks_ff{t}(eff_test_rows,:);
            elseif (ismember(method,[1,4]))
                output.X_train{ff,t} = X_train_tasks_ff{t}(eff_train_rows,[idx.ind_common,idx.ind_diff{t}]);
                output.X_test{ff,t} = X_test_tasks_ff{t}(eff_test_rows,[idx.ind_common,idx.ind_diff{t}]);         % use only available features
                output.prev_matrix_train{ff,t} = prev_matrix_train_tasks_ff{t}(eff_train_rows,[idx.ind_common,idx.ind_diff{t}]);
                output.prev_matrix_test{ff,t} = prev_matrix_test_tasks_ff{t}(eff_test_rows,[idx.ind_common,idx.ind_diff{t}]);
                output.next_matrix_train{ff,t} = next_matrix_train_tasks_ff{t}(eff_train_rows,[idx.ind_common,idx.ind_diff{t}]);
                output.next_matrix_test{ff,t} = next_matrix_test_tasks_ff{t}(eff_test_rows,[idx.ind_common,idx.ind_diff{t}]);
                output.Y_train{ff,t} = Y_train_tasks_ff{t}(eff_train_rows,:);
                output.Y_test{ff,t} = Y_test_tasks_ff{t}(eff_test_rows,:);
            elseif (ismember(method,[2,5,6,7,8]))
                output.X_train_common{ff,t} = X_train_tasks_ff{t}(eff_train_rows,idx.ind_common);
                output.X_test_common{ff,t} = X_test_tasks_ff{t}(eff_test_rows,idx.ind_common);         % use only available features
                output.X_train_diff{ff,t} = X_train_tasks_ff{t}(eff_train_rows,idx.ind_diff{t});
                output.X_test_diff{ff,t} = X_test_tasks_ff{t}(eff_test_rows,idx.ind_diff{t});         % use only available features
                output.prev_matrix_train_common{ff,t} = prev_matrix_train_tasks_ff{t}(eff_train_rows,idx.ind_common);
                output.prev_matrix_test_common{ff,t} = prev_matrix_test_tasks_ff{t}(eff_test_rows,idx.ind_common);
                output.prev_matrix_train_diff{ff,t} = prev_matrix_train_tasks_ff{t}(eff_train_rows,idx.ind_diff{t});
                output.prev_matrix_test_diff{ff,t} = prev_matrix_test_tasks_ff{t}(eff_test_rows,idx.ind_diff{t});
                output.next_matrix_train_common{ff,t} = next_matrix_train_tasks_ff{t}(eff_train_rows,idx.ind_common);
                output.next_matrix_test_common{ff,t} = next_matrix_test_tasks_ff{t}(eff_test_rows,idx.ind_common);
                output.next_matrix_train_diff{ff,t} = next_matrix_train_tasks_ff{t}(eff_train_rows,idx.ind_diff{t});
                output.next_matrix_test_diff{ff,t} = next_matrix_test_tasks_ff{t}(eff_test_rows,idx.ind_diff{t});
                output.Y_train{ff,t} = Y_train_tasks_ff{t}(eff_train_rows,:);
                output.Y_test{ff,t} = Y_test_tasks_ff{t}(eff_test_rows,:);
            end

            
            if (ismember(method,[1,2,4,5,6,7,8]))                   
                [output.prev_permute_train{ff,t},output.next_permute_train{ff,t},~,~] = seq_relation(output.ids_train{ff,t});
            end
            
        end
    end

end

