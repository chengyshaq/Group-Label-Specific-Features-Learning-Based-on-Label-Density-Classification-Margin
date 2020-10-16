function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels] = LDCMKELM (P, T, Pt, Tt, C,  gamma)
% LDCMKELM: Label-Density Classification Margin

T = T';
[n,nc] = size(T);     % nc ==> num_class n==> num_instance

%% TD 
PTDW = sum(T==1)/sum(sum(T==1))+1;
NTDW = sum(T==-1)/sum(sum(T==-1))+1;

PTT = T; PTT(PTT==-1) = 0;
PTD = PTT.*repmat(PTDW,n,1);

NTT = T; NTT(NTT==1) = 0;
NTD = NTT.*repmat(NTDW,n,1);

TD = PTD + NTD;

TT = TD;
%% KELM
Omega_train = kernel_matrix(P,gamma);
OutputWeight=((Omega_train+speye(n)/C)\TT);
Omega_test = kernel_matrix(P,gamma,Pt);
Outputs=(Omega_test' * OutputWeight)';

Pre_Labels = sign(Outputs);

HammingLoss=Hamming_loss(Pre_Labels,Tt);

RankingLoss=Ranking_loss(Outputs,Tt);
OneError=One_error(Outputs,Tt);
Coverage=coverage(Outputs,Tt);
Average_Precision=Average_precision(Outputs,Tt);