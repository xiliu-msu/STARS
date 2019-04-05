function [P,N,P_avlbl_ids,N_avlbl_ids] = seq_relation(ids)
    P_rm_inst = [];
    N_rm_inst = [];

    P = zeros(size(ids,1),size(ids,1));
    for i = 1:1:size(ids,1)
        xx = intersect(find(ids(:,1)==ids(i,1)), find(ids(:,2) == (ids(i,2)-1)));
        if (~isempty(xx))
            if (length(xx)==1)
                P(i,xx) = 1;
            else
                error('find duplicate previous second!!!');
            end
        else
            P(i,i) = 1;
            P_rm_inst = [P_rm_inst;i];
%             disp('cannot find the previous second!!!');
%             fprintf('%d,%d,%d\n',ids(i,1),ids(i,2),ids(i,3));
        end
    end
    P = sparse(P);
    
    
    
    N = zeros(size(ids,1),size(ids,1));
    for i = 1:1:size(ids,1)
        xx = intersect(find(ids(:,1)==ids(i,1)), find(ids(:,2) == (ids(i,2)+1)));
        if (~isempty(xx))
            if (length(xx)==1)
                N(i,xx) = 1;
            else
                error('find duplicate previous second!!!');
            end
        else
            N(i,i) = 1;
            N_rm_inst = [N_rm_inst;i];
%             disp('cannot find the previous second!!!');
%             fprintf('%d,%d,%d\n',ids(i,1),ids(i,2),ids(i,3));
        end
    end
    N = sparse(N);
    P_avlbl_inst = setdiff(1:1:size(ids,1),P_rm_inst);
    N_avlbl_inst = setdiff(1:1:size(ids,1),N_rm_inst);
    P_avlbl_ids = ids(P_avlbl_inst,:);
    N_avlbl_ids = ids(N_avlbl_inst,:);
end

