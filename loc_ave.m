clear;

sequences_train = sequences_str(1:1:10);


count_seq = 0;
for sequence = sequences_train
    count_seq = count_seq + 1;
    sequence{1}
    X = [];
    Y = [];
    annotation = importdata(['../public_data/train/',sequence{1},'/targets.csv']);
   
    
    loc_data = [];
    loc = importdata(['../public_data_preprocess/train/loc/loc_only_data/loc_',sequence{1},'.csv']);
    for line = 1:1:size(loc.data,1)
        v = (loc.data(line,1):0.001:loc.data(line,2))';
        room = zeros(1,9);
        room(loc.data(line,3)) = 1;
        loc_data = [loc_data;[v,repmat(room,length(v),1)]];
    end

    
    
    
    
    for i = 1:1:size(annotation.data,1)
        if (mod(i,100)==0)
            fprintf('.');
        end
        time_start = annotation.data(i,1);
        time_end = annotation.data(i,2);
        
        sec_time_loc =  loc_data(intersect(find(loc_data(:,1)>=time_start),find(loc_data(:,1)<time_end)),1);
        sec_loc = loc_data(intersect(find(loc_data(:,1)>=time_start),find(loc_data(:,1)<time_end)),2:end);
        if ~isempty(sec_loc)  
            % average
            ave_sec_loc = mean(sec_loc,1);
        else
            ave_sec_loc = zeros(1,9);   
        end
        
        %% unit time 
        X = [X;time_start,time_end,ave_sec_loc]; 
    end
     save(['../labeled_features/loc_ave/loc_ave_',sequence{1}],'X');
end


