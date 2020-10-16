function [OutputWeight,Omega_test,Y] = kelmtrain (P, T, Pt, C,  gamma)

n = size(T,2);
Omega_train = kernel_matrix(P,gamma);
OutputWeight=((Omega_train+speye(n)/C)\(T')); 

Y=(Omega_train' * OutputWeight)';                             
Omega_test = kernel_matrix(P,gamma,Pt);