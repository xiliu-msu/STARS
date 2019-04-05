
% clear;
% xx = load('./case_1_para_linear_slices_10_cv5.mat');
% para_rec = xx.para_rec_1;
% % output_struct.idx = feature_idx(1);
% best_para.fold = 1:1:5;
% best_para.room = 1:1:9;
% num_task  = 9;
% num_class = 20;
% 
% length_common = 71;
% length_diff = 101+70;
% 
% low_rank = 7 ;
% W_common = repmat({zeros(num_task,length_common)},length(best_para.fold),num_class);
% W_diff = repmat({[]},length(best_para.fold),num_task);
% U = repmat({rand(num_task,low_rank)},length(best_para.fold),num_class);
% V = repmat({rand(low_rank,length_common)},length(best_para.fold),num_class);
% L1 = repmat({[]},length(best_para.fold),num_task);
% L2 = repmat({[]},length(best_para.fold),num_task);
% residual = repmat({[]},length(best_para.fold),num_class);
% for ff = 1:1:length(best_para.fold)
%     for t = 1:1:length(best_para.room)
%         W_diff{ff,t} = para_rec{ff,t}.W(:,length_common+1:end);
%     end
%     for t = 1:1:length(best_para.room)
%         for k = 1:1:num_class
%             W_common{ff,k}(t,:) = para_rec{ff,t}.W(k,1:length_common);
%         end
%     end
%     
%     for r = 1:1:100
%         residule = norm(W_common{ff,11}-U{ff,11}*V{ff,11},'fro')^2;
%         fprintf('%d,%d,%.4f\n',ff,r,residule);
%         for k = 1:1:num_class
%             U{ff,k} = (W_common{ff,k}*V{ff,k}')*pinv(V{ff,k}*V{ff,k}');
%         end
%         for k = 1:1:num_class
%             V{ff,k} = pinv(U{ff,k}'*U{ff,k})*(U{ff,k}'*W_common{ff,k});
%         end
%     end
%     
%     for t = 1:1:length(best_para.room)
%         L1{ff,t} = para_rec{ff,t}.V1;
%         L2{ff,t} = para_rec{ff,t}.V2;
%     end
% 
% end
% ini_U = U;
% ini_V = V;
% ini_W_diff = W_diff;
% ini_L1 = L1;
% ini_L2 = L2;








clear;
data = load('./results/res_main1.mat');
para_rec = data.para_rec;
output_struct.idx = feature_idx(1);
best_para.fold = data.pp.fold;
best_para.room = data.pp.room;
num_task  = length(best_para.room);
num_class = 20;

low_rank = 8 ;
W_common = repmat({zeros(num_task,length(output_struct.idx.ind_common))},length(best_para.fold),num_class);
W_diff = repmat({[]},length(best_para.fold),num_task);
U = repmat({rand(num_task,low_rank)},length(best_para.fold),num_class);
V = repmat({rand(low_rank,length(output_struct.idx.ind_common))},length(best_para.fold),num_class);
L1 = repmat({[]},length(best_para.fold),num_task);
L2 = repmat({[]},length(best_para.fold),num_task);
residual = repmat({[]},length(best_para.fold),num_class);
for ff = 1:1:length(best_para.fold)
    for t = 1:1:length(best_para.room)
        W_diff{ff,t} = para_rec{ff,t}.W(:,length(output_struct.idx.ind_common)+1:end);
    end
    for t = 1:1:length(best_para.room)
        for k = 1:1:num_class
            W_common{ff,k}(t,:) = para_rec{ff,t}.W(k,1:length(output_struct.idx.ind_common));
        end
    end
    
    for r = 1:1:100
        residule = norm(W_common{ff,11}-U{ff,11}*V{ff,11},'fro')^2;
        fprintf('%d,%d,%.4f\n',ff,r,residule);
        for k = 1:1:num_class
            U{ff,k} = (W_common{ff,k}*V{ff,k}')*pinv(V{ff,k}*V{ff,k}');
        end
        for k = 1:1:num_class
            V{ff,k} = pinv(U{ff,k}'*U{ff,k})*(U{ff,k}'*W_common{ff,k});
        end
    end
    
    for t = 1:1:length(best_para.room)
        L1{ff,t} = para_rec{ff,t}.L1;
        L2{ff,t} = para_rec{ff,t}.L2;
    end

end
ini_U = U;
ini_V = V;
ini_W_diff = W_diff;
ini_L1 = L1;
ini_L2 = L2;


save (['./results/ini_UV_rank',num2str(low_rank)], 'ini_U', 'ini_V', 'ini_W_diff', 'ini_L1', 'ini_L2');
