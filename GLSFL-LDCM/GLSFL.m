function m = GLSFL(Y,K,epsilon)
% GLSFL Group-Label-Specific Features Learning by spectral clustering
%
%    Syntax
%
%       m = GLSFL( X,Y,alpha,epsilon,K )
%
%  [1] M. Belkin and P. Niyogi. Laplacian eigenmaps and spectral techniques 
%      for embedding and clustering. NIPS, 2001.

%% Construct affinity matrix
% Label similarity
A = 1-pdist(Y,'cosin');
% epsilon-eighborhoods
A(A<epsilon & A>0) = 0; A(A>-epsilon & A<0) = 0; A(isnan(A)) = 0;
% Affinity matrix
A = sparse(squareform(A));
A = abs(A);
%% Apply spectral clustering
% Compute degree matrix
num_label = size(A,1);
D = sum(A,2); D(D==0) = eps;
D = spdiags(1./sqrt(D),0,num_label,num_label);
% Compute Laplacian matrix(for largest eigenvectors)
L = D * A * D;
% Compute the eigenvectors
[U, ~] = eigs(L,K,'LM');
% Normalize the eigenvectors row-wise
U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
% Apply kmeans on U in row-wise
m = kmeans(U,K,'MaxIter',20,'OnlinePhase','off');

end

