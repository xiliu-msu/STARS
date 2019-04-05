function para = mtl_temporalloss_softmaxregress( test_see, C,X_common,X_diff,prev_matrix_common,prev_matrix_diff,next_matrix_common,next_matrix_diff,prev_permute,next_permute,flag_C,pp )
    num_class = size(C{1},2);
    num_task = length(X_common);
    num_total_instance = 0;
    for t = 1:1:num_task
        num_total_instance = num_total_instance + size(C{t},1);
    end
        
     % temporal consistency loss + weighted Single Task Softmax Regression (optimized by AGD with backtracking line search)
        % min_(w,v1,v2) sum_s sum_i sum_k -(C_ik*log P(y_i=k|x_i,x_(i-1),x_(i+1),w_sk,v_1skr,v_2skr))+ 
        % lamda* sum_s sum_k||w_sk|| + alpha*sum_s sum_k||v_sk|| + beta*||P-GP||,
        % where, w_k = uv, u is |s|xp, v is pxd 
        stpsz_U = repmat(0.001,num_class,1);
        stpsz_V = repmat(0.001,num_class,1);
        stpsz_W_diff = repmat(0.001,num_task,1);
        stpsz_L1 = repmat(0.001,num_task,1);
        stpsz_L2 = repmat(0.001,num_task,1);
        shrink_beta = 0.9;
        shrink_alpha = 0.5;
        stop_c = 0.05;
%         residual_old = 1000000000000000;
        if (~flag_C)
            [~, label] = max(C,[],2);
            temp_0and1s = zeros(num_sample, num_class);
            temp_0and1s(sub2ind(size(temp_0and1s),1:num_sample,label')) = 1;
            C = temp_0and1s;
        end
        X = cell(num_task,1);
        prev_matrix = cell(num_task,1);
        next_matrix = cell(num_task,1);
        for t = 1:1:num_task
            X{t} = [X_common{t},X_diff{t}];
            prev_matrix{t} = [prev_matrix_common{t},prev_matrix_diff{t}];
            next_matrix{t} = [next_matrix_common{t},next_matrix_diff{t}];
        end



        % initialization 
        U = pp.ini_U;
        V = pp.ini_V;
        W_diff = pp.ini_W_diff;
        L1 = pp.ini_L1;
        L2 = pp.ini_L2;
        f_old = mtl_temporalloss_softmaxregress_cost(C,X,prev_matrix,next_matrix,prev_permute,next_permute,U,V,W_diff,L1,L2,pp );
        U_mmt = U;
        U_old = U;
        U_new = U;
        V_mmt = V;
        V_old = V;
        V_new = V;
        W_diff_mmt = W_diff;
        W_diff_old = W_diff;
        W_diff_new = W_diff;
        L1_mmt = L1;
        L1_old = L1;
        L1_new = L1;
        L2_mmt = L2;
        L2_old = L2;
        L2_new = L2;


        num_iter = 100000;
        residual = zeros(num_iter,1);
        for r = 2:num_iter
            if (num_total_instance == 0 )
                break;
            end

%                   update U
            for k = 1:1:num_class
                U_old{k} = U{k};
                grad_f = mtl_temporalloss_softmaxregress_gd(C,X,X_common,X_diff,prev_matrix,prev_matrix_common,prev_matrix_diff,next_matrix,next_matrix_common,next_matrix_diff,prev_permute,next_permute,U_mmt,V,W_diff,L1,L2,pp,'U',k );   
                U_new{k} = U_mmt{k} - stpsz_U(k)*grad_f;
                % backtracking line search               
                f_new = mtl_temporalloss_softmaxregress_cost(C,X,prev_matrix,next_matrix,prev_permute,next_permute,U_new,V,W_diff,L1,L2,pp);
                if (f_new >(f_old - shrink_alpha*stpsz_U(k)*norm(grad_f,'fro')))
                    stpsz_U(k) = shrink_beta*stpsz_U(k);
                else
                    U{k} = U_new{k};
                    f_old = f_new;
                end
                % update U
                U_mmt{k} = U{k} + (r-1)/(r+2)*(U{k}-U_old{k});
            end



%                   update V
            for k = 1:1:num_class
                V_old{k} = V{k};
                grad_f = mtl_temporalloss_softmaxregress_gd(C,X,X_common,X_diff,prev_matrix,prev_matrix_common,prev_matrix_diff,next_matrix,next_matrix_common,next_matrix_diff,prev_permute,next_permute,U,V_mmt,W_diff,L1,L2,pp,'V',k );   
                V_new{k} = V_mmt{k} - stpsz_V(k)*grad_f;
                % backtracking line search               
                f_new = mtl_temporalloss_softmaxregress_cost(C,X,prev_matrix,next_matrix,prev_permute,next_permute,U,V_new,W_diff,L1,L2,pp);
                if (f_new >(f_old - shrink_alpha*stpsz_V(k)*norm(grad_f,'fro')))
                    stpsz_V(k) = shrink_beta*stpsz_V(k);
                else
                    V{k} = V_new{k};
                    f_old = f_new;
                end
                % update V
                V_mmt{k} = V{k} + (r-1)/(r+2)*(V{k}-V_old{k});
            end



%                   update W_diff
            for t = 1:1:num_task
                if (size(W_diff{t},2) == 0)
                    break;
                end
                W_diff_old{t} = W_diff{t};
                grad_f = mtl_temporalloss_softmaxregress_gd(C,X,X_common,X_diff,prev_matrix,prev_matrix_common,prev_matrix_diff,next_matrix,next_matrix_common,next_matrix_diff,prev_permute,next_permute,U,V,W_diff_mmt,L1,L2,pp,'W_diff',t );   
                W_diff_new{t} = W_diff_mmt{t} - stpsz_W_diff(t)*grad_f;
                % backtracking line search               
                f_new = mtl_temporalloss_softmaxregress_cost(C,X,prev_matrix,next_matrix,prev_permute,next_permute,U,V,W_diff_new,L1,L2,pp);
                if (f_new >(f_old - shrink_alpha*stpsz_W_diff(t)*norm(grad_f,'fro')))
                    stpsz_W_diff(t) = shrink_beta*stpsz_W_diff(t);
                else
                    W_diff{t} = W_diff_new{t};
                    f_old = f_new;
                end
                % update W_diff_mmt
                W_diff_mmt{t} = W_diff{t} + (r-1)/(r+2)*(W_diff{t}-W_diff_old{t});
            end


            % update L1
            for t = 1:1:num_task
                L1_old{t} = L1{t};
                grad_f = mtl_temporalloss_softmaxregress_gd(C,X,X_common,X_diff,prev_matrix,prev_matrix_common,prev_matrix_diff,next_matrix,next_matrix_common,next_matrix_diff,prev_permute,next_permute,U,V,W_diff,L1_mmt,L2,pp,'L1',t );   
                L1_new{t} = L1_mmt{t} - stpsz_L1(t)*grad_f;
                % backtracking line search               
                f_new = mtl_temporalloss_softmaxregress_cost(C,X,prev_matrix,next_matrix,prev_permute,next_permute,U,V,W_diff,L1_new,L2,pp);
                if (f_new >(f_old - shrink_alpha*stpsz_L1(t)*norm(grad_f,'fro')))
                    stpsz_L1(t) = shrink_beta*stpsz_L1(t);
                else
                    L1{t} = L1_new{t};
                    f_old = f_new;
                end
                % update W_mmt
                L1_mmt{t} = L1{t} + (r-1)/(r+2)*(L1{t}-L1_old{t});
            end


            % update L2
            for t = 1:1:num_task
                L2_old{t} = L2{t};
                grad_f = mtl_temporalloss_softmaxregress_gd(C,X,X_common,X_diff,prev_matrix,prev_matrix_common,prev_matrix_diff,next_matrix,next_matrix_common,next_matrix_diff,prev_permute,next_permute,U,V,W_diff,L1,L2_mmt,pp,'L2',t );   
                L2_new{t} = L2_mmt{t} - stpsz_L2(t)*grad_f;
                % backtracking line search               
                f_new = mtl_temporalloss_softmaxregress_cost(C,X,prev_matrix,next_matrix,prev_permute,next_permute,U,V,W_diff,L1,L2_new,pp);
                if (f_new >(f_old - shrink_alpha*stpsz_L2(t)*norm(grad_f,'fro')))
                    stpsz_L2(t) = shrink_beta*stpsz_L2(t);
                else
                    L2{t} = L2_new{t};
                    f_old = f_new;
                end
                % update W_mmt
                L2_mmt{t} = L2{t} + (r-1)/(r+2)*(L2{t}-L2_old{t});
            end



            [ W_common,W ] = combine_W( U,V,W_diff,'all_tasks',[] );

%                     est_train = cell(1,num_task);
            est_test = cell(1,num_task);
            for t = 1:1:length(pp.room)
%                         est_train{1,t} = softmax_fun([[X_common{t},X_diff{t}],[prev_matrix_common{t},prev_matrix_diff{t}]*W{t}',[next_matrix_common{t},next_matrix_diff{t}]*W{t}'],[W{t},L1{t},L2{t}]);
                est_test{1,t} = softmax_fun([[test_see.X_test_common{1,t},test_see.X_test_diff{1,t}],[test_see.prev_matrix_test_common{1,t},test_see.prev_matrix_test_diff{1,t}]*W{t}',[test_see.next_matrix_test_common{1,t},test_see.next_matrix_test_diff{1,t}]*W{t}'],[W{t},L1{t},L2{t}]);
            end

            % evaluate
%                     w_brier_score_train_record = cross_validation_evaluation_weighted_brier_score( est_train,C );
            w_brier_score_test_record = cross_validation_evaluation_weighted_brier_score( est_test,test_see.Y_test );




            residual(r-1) = f_old;
            if(mod(r,1)==0)
                fprintf('iter=%d, ',r);
%                         fprintf('%f\n', f_old);
%                         fprintf('%f\n', w_brier_score_train_record(end));
                fprintf('%f\n', w_brier_score_test_record(end));
            end

            % stop criteria
            ss = 50;
            if (r>ss)
                if(abs(residual(r-ss)-f_old)/(num_total_instance*num_class)<stop_c)
                    break;
                end
            end



        end        
        para.U = U;
        para.V = V;
        para.W_diff = W_diff;
        para.L1 = L1;
        para.L2 = L2;
        para.residual = residual;
        [ para.W_common,para.W ] = combine_W( para.U,para.V,para.W_diff,'all_tasks',[] );

end


