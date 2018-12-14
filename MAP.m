function [klas, Z] = MAP(PostProbs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [klas, Z] = MAP(PostProbs)
%   
% calculate a partition by applying the Maximum A Posteriori Bayes
% allocation rule
%
%
% Inputs : 
%   PostProbs, a matrix of dimensions [n x K] of the posterior
%  probabilities of a given sample of n observations arizing from K groups
%
% Outputs:
%   klas: a vector of n class labels (z_1, ...z_n) where z_i =k \in {1,...K}
%       klas(i) = arg   max (PostProbs(i,k)) , for all i=1,...,n
%                     1<=k<=K
%               = arg   max  p(zi=k|xi;theta)
%                     1<=k<=K
%               = arg   max  p(zi=k;theta)p(xi|zi=k;theta)/sum{l=1}^{K}p(zi=l;theta) p(xi|zi=l;theta)
%                     1<=k<=K
%
%
%       Z : Hard partition data matrix [nxK] with binary elements Zik such
%       that z_ik =1 iff z_i = k
%
%%%%%%%%%%%%%%%%%%%%%%%%% Faicel Chamroukhi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[n, K] = size(PostProbs);
% maximum a posteriori rule
[~,klas] = max(PostProbs,[],2);
partition_MAP = (klas*ones(1,K))==(ones(n,1)*[1:K]);
% double   
Z = double(partition_MAP);

