function [ matrix_train_tasks, matrix_test_tasks, target_train_tasks, target_test_tasks, task_prob_train_tasks, task_prob_test_tasks, ids_train_tasks, ids_test_tasks] = preprocess( matrix_train, matrix_test, target_train, target_test, task_prob_train, task_prob_test, ids_train, ids_test, num_task, ind_location, multi_flag, intp_flag )
   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input: 
%   matrix_train / matrix_test: num_matrix x {n x d} 
%   target_train / target_test: num_target x {n x k} 
%   task_prob_train / task_prob_test: n x num_task 
%   ids_train / ids_test: n x 3 
%   num_task: t 
%   ind_location
%   multi_flag
%   intp_flag

% output:
%   matrix_train_tasks / matrix_test_tasks: num_task x num_matrix x {n x d} 
%   target_train_tasks / target_test_tasks: num_task x num_target x {n x k} 
%   task_prob_train_tasks / task_prob_test_tasks: num_task x {n x 1} 
%   ids_train_tasks / ids_test_tasks: num_task x 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    num_matrix = length(matrix_train);
    if (length(matrix_test)~=num_matrix)
        error('There are not equal number of testing matrices than training matrices!!!');
    end
    num_target = length(target_train);
    if (length(target_test)~=num_target)
        error('There are not equal number of testing targets than training targets!!!');
    end

    if (multi_flag == 0) 
        % standardize the data
        matrix_train_tasks = cell(1,num_matrix);
        matrix_test_tasks = cell(1,num_matrix); 
        for m = 1:1:num_matrix
            X_train = matrix_train{m};
            X_test = matrix_test{m};
            X_all = [X_train;X_test];
            [temp,~] = mapminmax(X_all',-1,1);
            X_all = temp';
            if (strcmp(intp_flag, 'mean'))
                [X_all,~] = impute(X_all, 1:size(X_all,2));
            elseif (isnumeric(intp_flag))
                X_all(isnan(X_all)) = -intp_flag;
            elseif (strcmp(intp_flag, 'nan'))
                X_all = X_all;
            end
            size_train = size(X_train,1);
            X_train = X_all(1:size_train,:);
            X_test = X_all((size_train+1):end,:);
            matrix_train_tasks{m} = X_train;
            matrix_test_tasks{m} = X_test;
        end
        target_train_tasks = target_train;
        target_test_tasks = target_test;
        task_prob_train_tasks = task_prob_train;
        task_prob_test_tasks = task_prob_test;
        ids_train_tasks = ids_train;
        ids_test_tasks = ids_test;
        
    elseif (multi_flag == 1)
        % standardize the data
        matrix_train_tasks = cell(num_task,num_matrix);
        matrix_test_tasks = cell(num_task,num_matrix); 
        target_train_tasks = cell(num_task,num_target);
        target_test_tasks = cell(num_task,num_target);
        task_prob_train_tasks = cell(num_task,1);
        task_prob_test_tasks = cell(num_task,1);
        ids_train_tasks = cell(num_task,1);
        ids_test_tasks = cell(num_task,1);

        [~,max_ind] = max(task_prob_train,[],2);
        task_label_train = zeros(size(task_prob_train));
        for i = 1:1:length(max_ind)
            task_label_train(i,max_ind(i)) = 1;
        end
        [~,max_ind] = max(task_prob_test,[],2);
        task_label_test = zeros(size(task_prob_test));
        for i = 1:1:length(max_ind)
            task_label_test(i,max_ind(i)) = 1;
        end
        ind_room_train = cell(num_task,1);
        ind_room_test = cell(num_task,1);
        for t = 1:1:num_task
            ind_room_train{t} = find(task_label_train(:,t) == 1);
            ind_room_test{t} = find(task_label_test(:,t) == 1 );
            task_prob_train_tasks{t} = task_prob_train(ind_room_train{t},t);
            task_prob_test_tasks{t} = task_prob_test(ind_room_test{t},t);
            ids_train_tasks{t} = ids_train(ind_room_train{t},:);
            ids_test_tasks{t} = ids_test(ind_room_test{t},:);
        end

        for m = 1:1:num_target
            for t = 1:1:num_task
                target_train_tasks{t,m} = target_train{m}(ind_room_train{t},:);
                target_test_tasks{t,m} = target_test{m}(ind_room_test{t},:);
            end
        end

        for m = 1:1:num_matrix
            X_train = matrix_train{m};
            X_test = matrix_test{m};
            X_all = [X_train;X_test];
            [temp,~] = mapminmax(X_all',-1,1);
            X_all = temp';
            if (strcmp(intp_flag, 'mean'))
                [X_all,~] = impute(X_all, 1:size(X_all,2));
            elseif (isnumeric(intp_flag))
                X_all(isnan(X_all)) = -intp_flag;
            elseif (strcmp(intp_flag, 'nan'))
                X_all = X_all;
            end
            size_train = size(X_train,1);
            X_train = X_all(1:size_train,:);
            X_test = X_all((size_train+1):end,:);
            for t = 1:1:num_task
                matrix_train_tasks{t,m} = X_train(ind_room_train{t},:);
                matrix_test_tasks{t,m} = X_test(ind_room_test{t},:);
            end
        end
        
        
        
        
        
    elseif (multi_flag == 2)
        % standardize the data
        X_all = [X_train;X_test];
        [temp,~] = mapminmax(X_all',-1,1);
        X_all = temp';
        if (strcmp(intp_flag, 'mean'))
            [X_all,~] = impute(X_all, 1:size(X_all,2));
        elseif (isnumeric(intp_flag))
            X_all(isnan(X_all)) = -intp_flag;
        elseif (strcmp(intp_flag, 'nan'))
            X_all = X_all;
        end
        size_train = size(X_train,1);
        X_train = X_all(1:size_train,:);
        X_test = X_all((size_train+1):end,:);
        
        % divide into tasks
        X_train_tasks = cell(num_task,1);
        X_test_tasks = cell(num_task,1);
        Y_train_tasks = cell(num_task,1);
        Y_test_tasks = cell(num_task,1);
        task_prob_train_tasks = cell(num_task,1);
        task_prob_test_tasks = cell(num_task,1);
        ids_train_tasks = cell(num_task,1);
        ids_test_tasks = cell(num_task,1);
        
        pp.lamda = 0.01;
        loc_X_train = X_train(:,ind_location);
        loc_X_test = X_test(:,ind_location);
        [loc_X_train,loc_X_test,~,~,~] = RandomFeature_RBF1(loc_X_train, loc_X_test,5000, 1/size(loc_X_train,2));
        para = stl_softmaxregress_AGD( loc_X_train,task_prob_train,1,pp );
        est_task_prob_train = softmax_fun(loc_X_train,para.W);
        est_task_prob_test = softmax_fun(loc_X_test,para.W);
        fprintf('Room classification...\n');
        [~,~,~,~] = evaluation( task_prob_train, est_task_prob_train, 'matrix', num_task,0,0,0);
        [~,~,~,~] = evaluation( task_prob_test, est_task_prob_test, 'matrix', num_task,0,0,0);


        
        for t = 1:1:num_task
            X_train_tasks{t} = X_train;
            X_test_tasks{t} = X_test;
            Y_train_tasks{t} = Y_train;
            Y_test_tasks{t} = Y_test;
            task_prob_train_tasks{t} = est_task_prob_train(:,t);
            task_prob_test_tasks{t} = est_task_prob_test(:,t);
            ids_train_tasks{t} = ids_train;
            ids_test_tasks{t} = ids_test;
        end
    end
end

