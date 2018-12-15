function [y, states, Z, mean_function] = sample_hmmr(x, prior, trans_mat, betak, sigma2k)
%
% sample_hmmr sample observations from a Hidden Markov Model Regression
% with parameters prior, trans_mat, betak, sigma2k for the domain x
%
%
% Outpus:
%         y: the generated observations sequence
%         states: the corresponding generated state sequences
%         Z: a binary allocation matrix
%         mean_function: the non-noisy regression function
%
%
%
%
%
% Faicel Chamroukhi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = size(betak, 1)-1
n = length(x);
X = designmatrix(x, p); % design matrix
K = length(prior);

Z = zeros(n,K);
states = zeros(n,1);
mean_function= zeros(n,1);
y= zeros(n,1);

t=1;
pz1 = prior;
z1k = mnrnd(1,pz1);%tirage de z1 selon une multinomiale
z1 = find(z1k == 1);

beta_z1  =  betak(:,z1);
sigma_z1 = sqrt(sigma2k(z1));
r1 = X(1,:)';
E_y1 = beta_z1'*r1;
mean_function(1) = E_y1;
y(1) = normrnd(E_y1,sigma_z1);


Z(1,:) =z1k;
states(1)=z1;

zt_1= z1;
pzt_1 = pz1;

for t=2:n
    % sample the state z
    
    pzt = trans_mat(zt_1,:); % Pr( zt| zt_1)
    
    ztk = mnrnd(1,pzt);% sample zt given zt_1
    
    zt = find(ztk == 1);
    zt_1 = zt;
    
    % sample the regression model
    beta_zt  = betak(:,zt);
    sigma_zt = sqrt(sigma2k(zt));
    xt = X(t,:)';
    E_yt = beta_zt'*xt;
    mean_function(t) = E_yt;
    y(t) = normrnd(E_yt, sigma_zt);
    
    Z(t,:) =ztk;
    states(t)=zt;
end
