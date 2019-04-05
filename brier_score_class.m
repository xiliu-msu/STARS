function brier_score_class = brier_score_class( target, output)
   
    % brier scores
    residual_matrix = (target - output).^2;    
    unweight_residual_matrix = residual_matrix;
    brier_score_class = sum(unweight_residual_matrix,1)/size(residual_matrix,1);
  
end

