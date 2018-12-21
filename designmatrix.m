function X = designmatrix(x,p)
% function X = designmatrix(x,p)
% constructs the design matrix of a polynomial regression of degree p
%
%
% Faicel Chamroukhi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (size(x,2) ~= 1)
    x=x'; % a column vector
end
n = length(x);
X=zeros(n,p+1);
for i = 1:p+1
    X(:,i) = x.^(i-1);% X = [1 x x.^2 x.^3 x.^p]
end