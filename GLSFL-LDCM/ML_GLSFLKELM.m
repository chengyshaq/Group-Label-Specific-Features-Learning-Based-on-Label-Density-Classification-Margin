function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels] = ML_GLSFLKELM( train_data,train_target,test_data,test_target,opts )

%% Set parameters
cluster_size = opts.size;
epsilon      = opts.epsilon;
alpha        = opts.alpha;
mu          = opts.mu;
C          = opts.C;
gamma        = opts.gamma;
%% Get the size of data
num_label = size(train_target,1);
% num_test  = size(test_data,1);

%% Group-labels learning
K = ceil(num_label/min(cluster_size,num_label));
% K = cluster_size;
if K > 1
    m = GLSFL(train_target,K,epsilon);
else
    m = ones(num_label,1);
end

%% Specific features mining
Y = train_target;
Y(Y==-1) = 0;
V = GLSFL_LASSO(train_data,Y,K,m,alpha,mu);

%% Build classifier chains for each meta-label

for j = 1:K 
    idx_feature = (V(:,j)~=0); idx_meta = (m==j);
    meta_train_data = train_data(:,idx_feature);
    meta_test_data = test_data(:,idx_feature);
    meta_train_target = train_target(idx_meta,:)';
    meta_test_target = test_target(idx_meta,:)';
    [~,~,~,~,~,meta_Outputs,meta_Pre_Labels] = LDCMKELM ([train_data,meta_train_data], meta_train_target', [test_data,meta_test_data], meta_test_target',C,  gamma);     
    temp_Labels{j,1} = meta_Pre_Labels;
    temp_Outputs{j,1} = meta_Outputs;
    test_Labels{j,1} = meta_test_target';
end

Pre_Labels = cell2mat(temp_Labels);
Outputs = cell2mat(temp_Outputs);
Labels = cell2mat(test_Labels);

HammingLoss=Hamming_loss(Pre_Labels,Labels);

RankingLoss=Ranking_loss(Outputs,Labels);
OneError=One_error(Outputs,Labels);
Coverage=coverage(Outputs,Labels);
Average_Precision=Average_precision(Outputs,Labels);

end
