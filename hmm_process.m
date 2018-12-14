function Pz = hmm_process(loi_initiale,mat_trans,n)
%% Loi de la variable cachee pour un HMM p(z_1,...,z_n;pi,A) pour une
%% séquence donnée
%
%
%
%
%
% Faicel Chamroukhi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = length(loi_initiale);
Pz = zeros(n,K);

pz1 = loi_initiale;
Pz(1,:) = pz1;
for i=2:n
    pzi=[];
    pzi =  mat_trans'*Pz(i-1,:)';%p(z_i = k ) = sum_l (p(z_i=k,z_{i-1}=l)) = sum_l (p(z_i=k|z_i-1=l))*p(z_{i-1}= l) = sum_l A_{lk}*p(z_{i-1})
%     pzi = normalise(pzi);
    Pz(i,:) = pzi;
end