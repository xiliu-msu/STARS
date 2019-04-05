function  brier_score_class_record  = cross_validation_evaluation_brier_score_class( est,Y )
    
    [num_fold,num_task] = size(Y);

    % initilize the outputs
    if ( (num_fold > 1) && (num_task > 1) )
        brier_score_class_record = cell(num_fold+1,num_task+1);         
    elseif ( (num_fold == 1) && (num_task > 1) )
        brier_score_class_record = cell(num_fold,num_task+1);   % num_fold x num_task
    elseif ( (num_fold > 1) && (num_task == 1) )
        brier_score_class_record = cell(num_fold+1,num_task);   % num_fold x num_task
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
            brier_score_class_record{ff,t} = brier_score_class( Y{ff,t}, est{ff,t});
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
            brier_score_class_record{ff,end} = brier_score_class( Y_onefold{ff}, est_onefold{ff});
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
            brier_score_class_record{end,t} = brier_score_class( Y_onetask{t}, est_onetask{t});
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
        brier_score_class_record{end,end} = brier_score_class( Y_all, est_all);
    end
    
    
    
    
    
    
end

