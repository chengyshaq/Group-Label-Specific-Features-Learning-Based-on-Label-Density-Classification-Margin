clear;clc
addpath('Measures');
addpath('data');

load Image.mat;
rng(1);
% data = L2Norm(data);

%% Set parameters
opts.size     = 3;
opts.epsilon  = 1e-2;
opts.alpha    = 1;
opts.mu      = 1;
opts.C      = 2;
opts.gamma   = 5;
%% Train and Test
[M,N]=size(data);
indices=crossvalind('Kfold',M,10);
for i=1:10
    disp(i);
    test = (indices == i);
    train = ~test;
    train_data=data(train,:);
    train_target=target(:,train);
    test_data=data(test,:);
    test_target=target(:,test);
    [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels] = ...
        ML_GLSFLKELM( train_data,train_target,test_data,test_target,opts );
    HL(i) = HammingLoss;
    RL(i) = RankingLoss;
    OE(i) = OneError;
    CV(i) = Coverage;
    AP(i) = Average_Precision;
end