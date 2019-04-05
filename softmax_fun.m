function prob = softmax_fun( x,w )  % nxk
    % x is nxp
    % w is kxp
    
    num_class = size(w,1);
    temp = exp(x*w'); % nxk
    temp(isinf(temp)) = 1.0000e+256;
    temp_sum = sum(temp,2);  % nx1
    temp_sum = repmat(temp_sum,1,num_class); % nxk
    temp_sum(temp_sum == 0) = 1.0000e-256;
    prob = temp./temp_sum;  %nxk

%     prob = softmax(w*x')';
end

