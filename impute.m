function [ full_x,imp_v ] = impute( x, intp_col )
    full_x = x;
    imp_v = nan(size(x,2),1);
    for i = intp_col
        v =x(:,i);
        
        temp_col = v;
        temp_col(isnan(temp_col)) = 0;
        
        temp = find(~isnan(v));
        ave = sum(v(temp))/length(temp);
        
        temp = ave*isnan(v);
        
        temp_col = temp_col + temp;
        
        full_x(:,i) = temp_col;
        
        imp_v(i) = ave;
    end
end

