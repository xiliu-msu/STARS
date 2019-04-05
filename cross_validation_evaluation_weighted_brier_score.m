function [ w_brier_score_record ] = cross_validation_evaluation_weighted_brier_score( est,Y )
    
    [num_fold,num_task] = size(Y);

    % initilize the outputs
    if ( (num_fold > 1) && (num_task > 1) )
%         macro_f1_record = zeros(num_fold+1,num_task+1);    % num_fold x num_task
%         micro_f1_record = zeros(num_fold+1,num_task+1);    % num_fold x num_task   
%         brier_score_record = zeros(num_fold+1,num_task+1); % num_fold x num_task
        w_brier_score_record = zeros(num_fold+1,num_task+1);   % num_fold x num_task
    elseif ( (num_fold == 1) && (num_task > 1) )
%         macro_f1_record = zeros(num_fold,num_task+1);    % num_fold x num_task
%         micro_f1_record = zeros(num_fold,num_task+1);    % num_fold x num_task   
%         brier_score_record = zeros(num_fold,num_task+1); % num_fold x num_task
        w_brier_score_record = zeros(num_fold,num_task+1);   % num_fold x num_task
    elseif ( (num_fold > 1) && (num_task == 1) )
%         macro_f1_record = zeros(num_fold+1,num_task);    % num_fold x num_task
%         micro_f1_record = zeros(num_fold+1,num_task);    % num_fold x num_task   
%         brier_score_record = zeros(num_fold+1,num_task); % num_fold x num_task
        w_brier_score_record = zeros(num_fold+1,num_task);   % num_fold x num_task
    end
    
    %initialize the intermediate outputs   
    est_onefold = cell(num_fold,1);
    Y_onefold = cell(num_fold,1);
    est_onetask = cell(num_task,1);
    Y_onetask = cell(num_task,1);
    est_all = [];
    Y_all = [];
    
    % evaluate ff,t
    for ff = 1:1:num_fold
        for t = 1:1:num_task
            w_brier_score_record(ff,t) = weighted_brier_score_computation( Y{ff,t}, est{ff,t} );
        end
    end
    
    
    
     % evaluate ff,end
    if (num_task > 1)
        est_onefold = cell(num_fold,1);
        Y_onefold = cell(num_fold,1);  
        for ff = 1:1:num_fold
            for t = 1:1:num_task
                est_onefold{ff} = [est_onefold{ff};est{ff,t}];
                Y_onefold{ff} = [Y_onefold{ff};Y{ff,t}];
            end
            w_brier_score_record(ff,end) = weighted_brier_score_computation( Y_onefold{ff},est_onefold{ff} );
        end
    end
    
    % evaluate end,t
    if (num_fold > 1)
        est_onetask = cell(num_task,1);
        Y_onetask = cell(num_task,1);
        for t = 1:1:num_task
            for ff = 1:1:num_fold
                est_onetask{t} = [est_onetask{t};est{ff,t}];
                Y_onetask{t} = [Y_onetask{t};Y{ff,t}];
            end
            w_brier_score_record(end,t) = weighted_brier_score_computation( Y_onetask{t},est_onetask{t} );
        end
    end
    
    % evaluate all
    est_all = [];
    Y_all = [];
    if ( (num_fold > 1) && (num_task > 1) )
        for ff = 1:1:num_fold
            est_all = [est_all;est_onefold{ff}];
            Y_all = [Y_all;Y_onefold{ff}];
        end
        w_brier_score_record(end,end) = weighted_brier_score_computation( Y_all,est_all );
    end
    
    
    
    
    
    
end

