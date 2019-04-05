function grad_f = mtl_temporalloss_softmaxregress_gd( C,X,X_common,X_diff,prev_matrix,prev_matrix_common,prev_matrix_diff,next_matrix,next_matrix_common,next_matrix_diff,prev_permute,next_permute,U,V,W_diff,L1,L2,pp,var,var_idx )
    num_task = length(X_common);
    num_class = size(C{1},2);
    switch var 
        case 'U'    % k x( num_task x p)
            [ ~,W ] = combine_W( U,V,W_diff,'all_tasks',[] );
            prob = cell(num_task,1);
            for t = 1:1:num_task
                prob{t} = softmax_fun([X{t},prev_matrix{t}*W{t}',next_matrix{t}*W{t}'],[W{t},L1{t},L2{t}]);    % nxk
            end


            diff_prob = cell(num_task,1);
            l1_diff_prob = cell(num_task,1);
            l2_diff_prob = cell(num_task,1);
            for t = 1:1:num_task
                diff_prob{t} = (prob{t} - C{t})';
                l1_diff_prob{t} = ((prob{t}-C{t})*L1{t})';
                l2_diff_prob{t} = ((prob{t}-C{t})*L2{t})';
            end
            k = var_idx;
            grad_f = 2*pp.reg_U(k)*U{k};    % num_task x p for class k
            for t = 1:1:num_task
                temp1 = diff_prob{t}(k,:)*X_common{t};   % 1xd
                temp2 = l1_diff_prob{t}(k,:)*prev_matrix_common{t}; % 1xd
                temp3 = l2_diff_prob{t}(k,:)*next_matrix_common{t}; % 1xd
                temp_all = temp1+temp2+temp3;   % 1xd
                if (pp.reg_smooth(t) ~= 0)
                    num_sample = size(X_common{t},1);
                    diff = prob{t} - prev_permute{t}*prob{t};  % n_t x k
                    temp = zeros(num_sample,num_class);
                    temp(:,k) = ones(num_sample,1);   % n_t x k
                    reg_temp1 = prob{t}.*(temp - repmat(prob{t}(:,k),1,num_class));   % n_t x k
                    reg_temp2 = prob{t}.*(repmat(L1{t}(:,k)',num_sample,1) - prob{t}*repmat(L1{t}(:,k),1,num_class));   %n_t x k
                    reg_temp3 = prob{t}.*(repmat(L2{t}(:,k)',num_sample,1) - prob{t}*repmat(L2{t}(:,k),1,num_class));   %n_t x k
                    reg_prev_temp1 = prev_permute{t}*reg_temp1;
                    reg_prev_temp2 = prev_permute{t}*reg_temp2;
                    reg_prev_temp3 = prev_permute{t}*reg_temp3;                            
                    reg_term1 = sum(diff.*reg_temp1,2)';
                    reg_term2 = sum(diff.*reg_temp2,2)';
                    reg_term3 = sum(diff.*reg_temp3,2)';
                    reg_prev_term1 = sum(diff.*reg_prev_temp1,2)';  % 1xn_t
                    reg_prev_term2 = sum(diff.*reg_prev_temp2,2)';
                    reg_prev_term3 = sum(diff.*reg_prev_temp3,2)';

                    temptemp = (reg_term1 - reg_prev_term1*prev_permute{t})*X_common{t}...
                        + (reg_term2 - reg_prev_term2*prev_permute{t})*prev_matrix_common{t}...
                        + (reg_term3 - reg_prev_term3*prev_permute{t})*next_matrix_common{t};   % 1xd                        
                    temp_all = temp_all+2*pp.reg_smooth(t)*temptemp;   % 1xd
                end
                grad_f(t,:) = grad_f(t,:) + temp_all*V{k}'; % num_task x p
            end



        case 'V'    % k x ?d x p?
            [ ~,W ] = combine_W( U,V,W_diff,'all_tasks',[] );
            prob = cell(num_task,1);
            for t = 1:1:num_task
                prob{t} = softmax_fun([X{t},prev_matrix{t}*W{t}',next_matrix{t}*W{t}'],[W{t},L1{t},L2{t}]);    % nxk
            end

            diff_prob = cell(num_task,1);
            l1_diff_prob = cell(num_task,1);
            l2_diff_prob = cell(num_task,1);
            for t = 1:1:num_task
                diff_prob{t} = (prob{t} - C{t})';
                l1_diff_prob{t} = ((prob{t}-C{t})*L1{t})';
                l2_diff_prob{t} = ((prob{t}-C{t})*L2{t})';
            end                 
            k = var_idx;
            grad_f = 2*pp.reg_V(k)*V{k};    % pxd
            for t = 1:1:num_task
                temp1 = diff_prob{t}(k,:)*X_common{t};  % 1xd
                temp2 = l1_diff_prob{t}(k,:)*prev_matrix_common{t};  % 1xd
                temp3 = l2_diff_prob{t}(k,:)*next_matrix_common{t};  % 1xd                    
                temp_all = temp1+temp2+temp3;
                if (pp.reg_smooth(t) ~= 0)
                    num_sample = size(X_common{t},1);
                    diff = prob{t} - prev_permute{t}*prob{t};  % n_t x k
                    temp = zeros(num_sample,num_class);
                    temp(:,k) = ones(num_sample,1);   % n_t x k
                    reg_temp1 = prob{t}.*(temp - repmat(prob{t}(:,k),1,num_class));   % n_t x k
                    reg_temp2 = prob{t}.*(repmat(L1{t}(:,k)',num_sample,1) - prob{t}*repmat(L1{t}(:,k),1,num_class));   %n_t x k
                    reg_temp3 = prob{t}.*(repmat(L2{t}(:,k)',num_sample,1) - prob{t}*repmat(L2{t}(:,k),1,num_class));   %n_t x k
                    reg_prev_temp1 = prev_permute{t}*reg_temp1;
                    reg_prev_temp2 = prev_permute{t}*reg_temp2;
                    reg_prev_temp3 = prev_permute{t}*reg_temp3;                            
                    reg_term1 = sum(diff.*reg_temp1,2)';
                    reg_term2 = sum(diff.*reg_temp2,2)';
                    reg_term3 = sum(diff.*reg_temp3,2)';
                    reg_prev_term1 = sum(diff.*reg_prev_temp1,2)';  % 1xn_t
                    reg_prev_term2 = sum(diff.*reg_prev_temp2,2)';
                    reg_prev_term3 = sum(diff.*reg_prev_temp3,2)';

                    temptemp = (reg_term1 - reg_prev_term1*prev_permute{t})*X_common{t}...
                        + (reg_term2 - reg_prev_term2*prev_permute{t})*prev_matrix_common{t}...
                        + (reg_term3 - reg_prev_term3*prev_permute{t})*next_matrix_common{t};   % 1xd                        
                    temp_all = temp_all+2*pp.reg_smooth(t)*temptemp;   % 1xd
                end  
                grad_f = grad_f + U{k}(t,:)'*temp_all; % pxd
            end




        case 'W_diff'    % num_task x (kxd)
            t = var_idx;
            [ ~,W_t ] = combine_W( U,V,W_diff,'single_task',t );
            prob_t = softmax_fun([X{t},prev_matrix{t}*W_t',next_matrix{t}*W_t'],[W_t,L1{t},L2{t}]);    % nxk

            if (isempty(X_diff{t}))
                grad_f = [];
                return;
            else
                temp1 = (prob_t'-C{t}')*X_diff{t}; % kxd
                temp2 = ((prob_t-C{t})*L1{t})'*prev_matrix_diff{t};
                temp3 = ((prob_t-C{t})*L2{t})'*next_matrix_diff{t};   % kxd
                grad_f = temp1 + temp2 + temp3 + 2*pp.reg_W_diff(t)*W_diff{t}; %kxd

                if (pp.reg_smooth(t)~=0)
                    num_sample = size(X_diff{t},1);
                    diff = prob_t - prev_permute{t}*prob_t;  % nxk
                    reg_term1 = zeros(num_class,num_sample);
                    reg_term2 = zeros(num_class,num_sample);
                    reg_term3 = zeros(num_class,num_sample);
                    reg_prev_term1 = zeros(num_class,num_sample);
                    reg_prev_term2 = zeros(num_class,num_sample);
                    reg_prev_term3 = zeros(num_class,num_sample);                     
                    for q = 1:1:num_class
                        temp = zeros(num_sample,num_class);
                        temp(:,q) = ones(num_sample,1);   % nxk
                        reg_temp1 = prob_t.*(temp - repmat(prob_t(:,q),1,num_class));   % nxk
                        reg_temp2 = prob_t.*(repmat(L1{t}(:,q)',num_sample,1) - prob_t*repmat(L1{t}(:,q),1,num_class));   %nxk
                        reg_temp3 = prob_t.*(repmat(L2{t}(:,q)',num_sample,1) - prob_t*repmat(L2{t}(:,q),1,num_class));   %nxk
                        reg_prev_temp1 = prev_permute{t}*reg_temp1;
                        reg_prev_temp2 = prev_permute{t}*reg_temp2;
                        reg_prev_temp3 = prev_permute{t}*reg_temp3;                            
                        reg_term1(q,:) = sum(diff.*reg_temp1,2)';
                        reg_term2(q,:) = sum(diff.*reg_temp2,2)';
                        reg_term3(q,:) = sum(diff.*reg_temp3,2)';
                        reg_prev_term1(q,:) = sum(diff.*reg_prev_temp1,2)';
                        reg_prev_term2(q,:) = sum(diff.*reg_prev_temp2,2)';
                        reg_prev_term3(q,:) = sum(diff.*reg_prev_temp3,2)';
                    end 
                    temptemp = (reg_term1 - reg_prev_term1*prev_permute{t})*X_diff{t}...
                        + (reg_term2 - reg_prev_term2*prev_permute{t})*prev_matrix_diff{t}...
                        + (reg_term3 - reg_prev_term3*prev_permute{t})*next_matrix_diff{t};   % kxd
                    grad_f = grad_f + 2*pp.reg_smooth(t)*temptemp; %kxd
                end
            end






        case 'L1'   % num_task x kxk
            t = var_idx;
            [ ~,W_t ] = combine_W( U,V,W_diff,'single_task',t );
            prob_t = softmax_fun([X{t},prev_matrix{t}*W_t',next_matrix{t}*W_t'],[W_t,L1{t},L2{t}]);    % nxk

            prev_pred = prev_matrix{t}*W_t';
            grad_f = (prob_t'-C{t}')*prev_pred + 2*pp.reg_L1(t)*L1{t};  % kxk                
            if (pp.reg_smooth(t)~=0)
                num_sample = size(X{t},1);
                diff = prob_t - prev_permute{t}*prob_t;  % nxk      
                reg_term = zeros(num_class,num_sample);
                reg_prev_term = zeros(num_class,num_sample);
                for q = 1:1:num_class
                    temp = zeros(num_sample,num_class);
                    temp(:,q) = ones(num_sample,1);   % nxk
                    reg_temp = prob_t.*(temp-repmat(prob_t(:,q),1,num_class)); % nxk
                    reg_prev_temp = prev_permute{t}*reg_temp;
                    reg_term(q,:) =  sum(diff.*reg_temp,2)';
                    reg_prev_term(q,:) = sum(diff.*reg_prev_temp,2)';
                end
                temptemp = (reg_term-reg_prev_term*prev_permute{t})*prev_pred;
                grad_f = grad_f + 2*pp.reg_smooth(t)*temptemp;  % kxk
            end


        case 'L2'   % num_task x kxk
            t = var_idx;
            [ ~,W_t ] = combine_W( U,V,W_diff,'single_task',t );
            prob_t = softmax_fun([X{t},prev_matrix{t}*W_t',next_matrix{t}*W_t'],[W_t,L1{t},L2{t}]);    % nxk

            next_pred = next_matrix{t}*W_t';
            grad_f = (prob_t'-C{t}')*next_pred + 2*pp.reg_L1(t)*L1{t};  % kxk        
            if (pp.reg_smooth(t)~=0)
                num_sample = size(X{t},1);
                diff = prob_t - prev_permute{t}*prob_t;  % nxk      
                reg_term = zeros(num_class,num_sample);
                reg_prev_term = zeros(num_class,num_sample);
                for q = 1:1:num_class
                    temp = zeros(num_sample,num_class);
                    temp(:,q) = ones(num_sample,1);   % nxk
                    reg_temp = prob_t.*(temp-repmat(prob_t(:,q),1,num_class)); % nxk
                    reg_prev_temp = prev_permute{t}*reg_temp;
                    reg_term(q,:) =  sum(diff.*reg_temp,2)';
                    reg_prev_term(q,:) = sum(diff.*reg_prev_temp,2)';
                end
                temptemp = (reg_term-reg_prev_term*prev_permute{t})*next_pred;
                grad_f = grad_f + 2*pp.reg_smooth(t)*temptemp;  % kxk
            end
    end

end

