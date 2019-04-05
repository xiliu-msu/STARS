function w_brier_score = weighted_brier_score_computation( target, output)


    if ( (size(target,1)~=size(output,1)) || (size(target,2)~=size(output,2)) )
        error('target and output dimension do not agree!!!');
    end

    % brier scores
    residual_matrix = (target - output).^2; 
    wi = load('../data/class_weights.json');
    weight_residual_matrix = repmat(wi',size(residual_matrix,1),1).*residual_matrix;
    w_brier_score = sum(weight_residual_matrix(:))/size(residual_matrix,1);   
%     fprintf('w_brier_score=%f\n',w_brier_score);
  
end