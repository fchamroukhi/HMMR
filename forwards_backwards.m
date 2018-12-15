function [tau_tk, xi_tlk, alpha_tk, beta_tk, loglik] = forwards_backwards(prior, transmat, f_tk) 
%[tau_tk, xi_ikl, alpha, beta, loglik] = forwards_backwards(prior, transmat, fik, filter_only) 
% forwards_backwards : calculates the E-step of the EM algorithm for an HMM
% (Gaussian HMM)

% Inputs :
%
%         prior(k) = Pr(z_1 = k)
%         transmat(\ell,k) = Pr(z_t=k | z_{t-1} = \ell)
%         f_tk(t,k) = Pr(y_t | z_y=k;\theta) %gaussian
%
% Outputs:
%
%        tau_tk(t,k) = Pr(z_t=k | X): post probs (smoothing probs)
%        xi_tk\elll(t,k,\ell)  = Pr(z_t=k, z_{t-1}=\ell | Y) t =2,..,n
%        with Y = (y_1,...,y_n);
%        alpha_tk: [nxK], forwards probs: Pr(y1...yt,zt=k)
%        beta_tk: [nxK], backwards probs: Pr(yt+1...yn|zt=k)
%
%
%
% Faicel Chamroukhi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[T, K] = size(f_tk);

if nargin < 6, filter_only = 0; end

scale = ones(1,T);%pour que loglik = sum(log(scale)) part de zero

prior = prior(:); 
tau_tk = zeros(T,K);
xi_tlk = zeros(T-1,K,K);
alpha_tk = zeros(T,K);
beta_tk = zeros(T,K);

%% forwards: calculation of alpha_tk
t = 1;
alpha_tk(1,:) = prior' .* f_tk(t,:);
[alpha_tk(t,:), scale(t)] = normalise(alpha_tk(t,:));
for t=2:T
    [alpha_tk(t,:), scale(t)] = normalise((alpha_tk(t-1,:)*transmat) .* f_tk(t,:));
    %filtered_prob (t-1,:,:)= normalise((alpha(:,t-1) * fik(:,t)') .*transmat);
end
%%loglikehood (with the scaling technique) (see Rabiner's paper/book)
loglik = sum(log(scale));

if filter_only
  beta_tk = [];
  xi_tlk = alpha_tk;
  return;
end
%% backwards: calculation of beta_tk, tau_tk (and xi_tkl)
%t=T;
beta_tk(T,:) = ones(1,K);
tau_tk(T,:) = normalise(alpha_tk(T,:) .* beta_tk(T,:));
for t=T-1:-1:1
    beta_tk(t,:) =  normalise(  transmat * (beta_tk(t+1,:) .* f_tk(t+1,:))' );
    % transmat * (beta(t+1,:) .* fik(t+1,:))' /scale(t); 
    tau_tk(t,:) = normalise(alpha_tk(t,:) .* beta_tk(t,:));
    xi_tlk(t,:,:) = normalise((transmat .* (alpha_tk(t,:)' * (beta_tk(t+1,:) .* f_tk(t+1,:))))); 
end




