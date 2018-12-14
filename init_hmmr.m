%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hmmr =  init_hmmr(X, y, K, type_variance, EM_try)
% init_hmm_regression estime les paramètres initiaux d'un modèle de regression
% à processus markovien cache où la loi conditionnelle des observations est une gaussienne
%
% Entrees :
%       
%        data(i,:,nsignal) = x(i) : observation à l'instant i du signal
%        (séquence) nsignal (notez que pour la partie parametrisation des 
%        signaux les observations sont monodimentionnelles)
%        K : nbre d'états (classes) cachés
%        duree_signal :  duree du signal en secondes
%        fs : fréquence d'échantiloonnage des signaux en Hz
%        p : ordre de regression polynomiale
%
% Sorties :
%
%         hmmr_init : parametres initiaux du modele. structure
%         contenant les champs:
%         * le HMM initial          
%         1. prior (k) = Pr(Z(1) = k) avec k=1,...,K. loi initiale de z.
%         2. trans_mat(\ell,k) = Pr(z(i)=k | z(i-1)=\ell) : matrice des transitions
%         *         
%         3. regression_params initial : structure contenant les parametres
%         initiaux du modele de regression
%         3.1. betak : le vecteur parametre de regression associe a la classe k.
%         vecteur colonne de dim [(p+1)x1]
%         3.2 sigma2k(k) = variance de x(i) sachant z(i)=k; sigma2k(j) =
%         sigma^2_k.
%         mu(:,k) = Esperance de x(i) sachant z(i) = k ; mu(:,k) =
%         beta_k'ri où ri est un vecteur colonne de covariates:
%         ri=[1,t_i,...,(t_i)^p]' avec p l'ordre de regression.
%*        autres champs :
%         3.3. f_tk : les probabilités des observations. f_tk f(xi|z_i=k;theta)
%              = N(x_i;betak'ri,sigmak^2). martice de dim [nxk]
%         3.4 log_f_tk : logarithme néperien de f_tk.
%         3.5 phi = [1 t1 t1^2 ... t1^q
%                    1 t2 t2^2 ... t2^q
%                          ..
%                    1 ti ti^2 ... ti^q
%                          ..
%                    1 tn tn^2 ... tn^q]
%
%
% Faicel Chamroukhi, Novembre 2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
if strcmp(type_variance,'homoskedastic')
    homoskedastic = 1;
else
    homoskedastic = 0;
end
m = length(y);
%% Tnitialisation en tenant compte de la contrainte:

% Initialisation de la matrice des transitions
Mask = .5*eye(K);%masque d'ordre 1
for k=1:K-1
    ind = find(Mask(k,:) ~= 0);
    Mask(k,ind+1) = .5;
end
hmmr.trans_mat = Mask;

% Initialisation de la loi initiale de la variable cachee
hmmr.prior = [1;zeros(K-1,1)];
hmmr.stats.Mask = Mask;
%  Initialisation des coeffecients de regression et des variances.

hmmr_reg = init_hmmr_regressors(X, y, K, type_variance, EM_try);

hmmr.reg_param = hmmr_reg;

% % % initial loglik values
% log_f_tk = zeros(m, K);
% for k = 1:K
%     muk = X*hmmr_reg.betak(:,k);
%     if homoskedastic
%        sigma2k = hmmr_reg.sigma2;
%     else
%         sigma2k = hmmr_reg.sigma2k(k);
%     end
%     z=((y-muk).^2)/sigma2k;
%     log_f_tk(:,k) =  -0.5*(log(2*pi)+log(sigma2k)) - 0.5*z;%log(gaussian )
% end
% 
% log_f_tk = min(log_f_tk,log(realmax));
% log_f_tk = max(log_f_tk ,log(realmin));
% 
% f_tk = exp(log_f_tk);
% 
% hmmr.stats.f_tk = f_tk;
% hmmr.stats.log_f_tk = log_f_tk;
% hmmr.stats.muk = muk;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
function hmmr_reg_param = init_hmmr_regressors(X, y, K, type_variance, EM_try)

if strcmp(type_variance,'homoskedastic')
    homoskedastic = 1;
else
    homoskedastic = 0;
end
[m, P] = size(X);
%m = length(y);


if  EM_try ==1% uniform segmentation into K contiguous segments, and then a regression
    zi = round(m/K)-1;
    for k=1:K
        yk = y((k-1)*zi+1:k*zi);
        Xk = X((k-1)*zi+1:k*zi,:);

        hmmr_reg_param.betak(:,k) = inv(Xk'*Xk)*Xk'*yk;%regress(yk,Xk); % for a use in octave, where regress doesnt exist
        
        if homoskedastic
            hmmr_reg_param.sigma2 = var(y);
        else
            % muk = Xk*hmmr_reg_param.betak(:,k);
            % sigma2k ((yk-muk)'*(yk-muk))/zi;%
            sigma2k = var(yk);
            hmmr_reg_param.sigma2k(k) =  sigma2k;
        end
    end
    
else % random segmentation into contiguous segments, and then a regression
    Lmin= P+1;%minimum length of a segment %10
    tk_init = zeros(K,1);
    tk_init(1) = 0;
    K_1=K;
    for k = 2:K
        K_1 = K_1-1;
        temp = tk_init(k-1)+Lmin:m-K_1*Lmin;
        ind = randperm(length(temp));
        tk_init(k)= temp(ind(1));
    end
    tk_init(K+1) = m;
    for k=1:K
        i = tk_init(k)+1;
        j = tk_init(k+1);
        yk = y(i:j);%y((k-1)*zi+1:k*zi);
        Xk = X(i:j,:);%X((k-1)*zi+1:k*zi,:);
        hmmr_reg_param.betak(:,k) = inv(Xk'*Xk)*Xk'*yk;%regress(yk,Xk); % for a use in octave, where regress doesnt exist
        
        if homoskedastic
            hmmr_reg_param.sigma2 = var(y);
        else
            muk = Xk* hmmr_reg_param.betak(:,k);
            hmmr_reg_param.sigma2k(k) = ((yk-muk)'*(yk-muk))/length(yk);%
            %hmmr_reg_param.sigma2k(k) =  1;
        end
    end  
 end
%

