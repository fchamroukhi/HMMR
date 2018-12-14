    function [y, klas, Z, mean_function] = sample_hmmr(loi_initiale,mat_trans,betak,sigma2k, X)
%
%
%
%
%
%
%
%
%
%
% X: design matrix of a p-degree polynomial regression
%
%
% Faicel Chamroukhi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[n, P]=size(X);
K = length(loi_initiale);

Z = zeros(n,K);
klas = zeros(n,1);
mean_function= zeros(n,1);
y= zeros(n,1);

t=1;
pz1 = loi_initiale;
z1k = mnrnd(1,pz1);%tirage de z1 selon une multinomiale
z1 = find(z1k == 1);

beta_z1  =  betak(:,z1);
sigma_z1 = sqrt(sigma2k(z1));
r1 = X(1,:)';
E_y1 = beta_z1'*r1;
mean_function(1) = E_y1;
y(1) = normrnd(E_y1,sigma_z1);  


Z(1,:) =z1k;
klas(1)=z1;

zt_1= z1;
pzt_1 = pz1;

for t=2:n
    % sim de la sequence z
    
    pzt = mat_trans(zt_1,:); % proba de zt sachant zt_1   
%     pzt = normalise(pzt);

    ztk = mnrnd(1,pzt);% tirage de zt sachant zt_1

    zt = find(ztk == 1);
    zt_1 = zt;  

    % les param de regression
    beta_zt  = betak(:,zt);
    sigma_zt = sqrt(sigma2k(zt));
    xt = X(t,:)';
    E_yt = beta_zt'*xt;
    mean_function(t) = E_yt;
    y(t) = normrnd(E_yt, sigma_zt); 
    
    Z(t,:) =ztk;
    klas(t)=zt;
end
