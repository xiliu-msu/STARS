function f = mtl_temporalloss_softmaxregress_cost( C,X,prev_matrix,next_matrix,prev_permute,next_permute,U,V,W_diff,L1,L2,pp )
    num_class = size(C{1},2);
    num_task = length(X);

    f = 0;
    [ ~,W ] = combine_W( U,V,W_diff,'all_tasks',[] );

    for k = 1:1:num_class
        f = f + pp.reg_U(k)*norm(U{k},'fro')^2 + pp.reg_V(k)*norm(V{k},'fro')^2;
    end
    for t = 1:1:num_task   
        prob_t = softmax_fun([X{t},prev_matrix{t}*W{t}',next_matrix{t}*W{t}'],[W{t},L1{t},L2{t}]);    % nxk
        prob_t(prob_t==0) = 1.0000e-256;
        temp = -C{t}.*log(prob_t);   % nxk
        f = f + sum(temp(:)) + pp.reg_W_diff(t)*norm(W_diff{t},'fro')^2 + pp.reg_L1(t)*norm(L1{t},'fro')^2 + pp.reg_L2(t)*norm(L2{t},'fro')^2 + pp.reg_smooth(t)*norm(prob_t-prev_permute{t}*prob_t,'fro')^2;
    end
            
            
end

