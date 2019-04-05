function [ W_common,W ] = combine_W( U,V,W_diff,ss,task )
    num_class = length(U);
    switch ss
        case 'all_tasks'
            if (~isempty(task))
                error('the task variable should be empty !!!');
            end
            
            num_task = length(W_diff);

            W_temp = cell(num_class,1);
            for k = 1:1:num_class
                W_temp{k} = U{k}*V{k};    % num_task x D
            end          
            W_common = repmat({[]},num_task,1);
            for t = 1:1:num_task
                for k = 1:1:num_class
                    W_common{t} = [W_common{t};W_temp{k}(t,:)];
                end
            end
            W = repmat({[]},num_task,1);
            for t = 1:1:num_task
                W{t} = [W_common{t},W_diff{t}];
            end
        case 'single_task'
            if (~isscalar(task))
                error('the task variable should be a scalar !!!');
            end
            
            num_common_feature = size(V{1},2);

            W_common = zeros(num_class,num_common_feature);
            for k = 1:1:num_class
                W_common(k,:) = U{k}(task,:)*V{k};    % 1 x D
            end          
            
            W = [W_common,W_diff{task}];
    end

end

