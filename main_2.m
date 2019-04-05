clear;

method = 2;
filename = '/../data/slices_10_cv5.mat';
num_class = 20;
flag_nonlinear = 0;
flag_smalldata = 0;
low_rank = 8;
ini_filename = ['./ini_UV_rank',num2str(low_rank),'.mat'];
idx = feature_idx(1);


pp.fold = [1,2,3,4,5];
pp.room = [1,2,3,4,5,6,7,8,9];
pp.reg_U = ones(1,num_class);
pp.reg_V = zeros(1,num_class);
pp.reg_W_diff = ones(1,length(pp.room));
pp.reg_L1 = [4,2,4,4,2,2,2,1,8];
pp.reg_L2 = [4,2,4,4,2,2,2,1,8];
pp.reg_smooth = 0.5*ones(1,9);
pp.ini_filename = ini_filename;


% preprocess data
data = cross_validation_preprocess( method,filename,idx,length(pp.room) );
fprintf('processing finished!\n\n');
 
% training

% main function
[performance, para_rec, pp] = smarthome( data,pp,num_class );
