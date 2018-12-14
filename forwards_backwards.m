function [tik, xi_ilk, alpha, beta, loglik] = forwards_backwards(prior, transmat, fik) 
%[tik, xi_ikl, alpha, beta, loglik] = forwards_backwards(prior, transmat, fik, filter_only) 
% forwards_backwards calcul des # probas pour un HMM avec les procèdures
% forwards backwards.

% Entrees :
%
%         prior(k) = Pr(z_1 = k)
%         transmat(\ell,k) = Pr(z_i=k | z_{i-1} = \ell)
%         fik(i,k) = Pr(x_i | z_i=k;\theta) %gaussienne
%
% Sorties:
%
%        tik(i,k) = Pr(z_i=k | X) 
%        xi_ik\elll(i,k,\ell)  = Pr(z_i=k, z_{i-1}=\ell | X) i =2,..,n
%        avec X = (x_1,...,x_n);
%
%
%
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fik = fik';

T = size(fik, 1);

if nargin < 6, filter_only = 0; end

K = length(prior);
scale = ones(1,T);%pour que loglik = sum(log(scale)) part de zero

prior = prior(:); 
tik = zeros(T,K);
xi_ilk = zeros(T-1,K,K);
alpha = zeros(T,K);
beta = zeros(T,K);

%% calcul des alphaik et tik (et xiikl)
t = 1;
alpha(1,:) = prior' .* fik(t,:);
[alpha(t,:), scale(t)] = normalise(alpha(t,:));

for t=2:T
    [alpha(t,:), scale(t)] = normalise((alpha(t-1,:)*transmat) .* fik(t,:));
    %filtered_prob (t-1,:,:)= normalise((alpha(:,t-1) * fik(:,t)') .*transmat);
end
%%loglik (technique du scaling)
loglik = sum(log(scale));

if filter_only
  beta = [];
  xi_ilk = alpha;
  return;
end
%% calcul des betaik et tik (et xiikl)
%t=T;
beta(T,:) = ones(1,K);
tik(T,:) = normalise(alpha(T,:) .* beta(T,:));
for t=T-1:-1:1
    beta(t,:) =  normalise(  transmat * (beta(t+1,:) .* fik(t+1,:))' );
    % transmat * (beta(t+1,:) .* fik(t+1,:))' /scale(t); 
    tik(t,:) = normalise(alpha(t,:) .* beta(t,:));
    xi_ilk(t,:,:) = normalise((transmat .* (alpha(t,:)' * (beta(t+1,:) .* fik(t+1,:))))); 
end




