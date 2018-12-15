function hmmr = learn_hmmr(x, y, K, p,...
    type_variance, total_EM_tries, max_iter_EM, threshold, verbose)

% learn_hmmr learn a Regression model with a Hidden Markov Process (HMMR)
% for modeling and segmentation of a time series with regime changes.
% The learning is performed by the EM (Baum-Welch) algorithm.
%
%
% Inputs :
%
%          (x,y) : a time series composed of m points : dim(y)=[m 1]
%                * Each curve is observed during the interval [0,T], i.e x =[t_1,...,t_m]
%
%           K : Number of polynomial regression components (regimes)
%          	p : degree of the polynomials
%
% Outputs :
%
%         hmmr: the estimated HMMR model. a structure composed of:
%
%         prior: [Kx1]: prior(k) = Pr(z_1=k), k=1...K
%         trans_mat: [KxK], trans_mat(\ell,k) = Pr(z_t = k|z_{t-1}=\ell)
%         reg_param: the paramters of the regressors:
%                 betak: regression coefficients
%                 sigma2k (or sigma2) : the variance(s)
%         Stats:
%           tau_tk: smoothing probs: [nxK], tau_tk(t,k) = Pr(z_i=k | y1...yn)
%           alpha_tk: [nxK], forwards probs: Pr(y1...yt,zt=k)
%           beta_tk: [nxK], backwards probs: Pr(yt+1...yn|zt=k)
%           xi_tkl: [(n-1)xKxK], joint post probs : xi_tk\elll(t,k,\ell)  = Pr(z_t=k, z_{t-1}=\ell | Y) t =2,..,n
%           X: [nx(p+1)] regression design matrix
%           nu: model complexity
%           parameter_vector
%           f_tk: [nxK] f(yt|zt=k)
%           log_f_tk: [nxK] log(f(yt|zt=k))
%           loglik: log-likelihood at convergence
%           stored_loglik: stored log-likelihood values during EM
%           cputime: for the best run
%           cputime_total: for all the EM runs
%           klas: [nx1 double]
%           Zik: [nxK]
%           state_probs: [nxK]
%           BIC: -2.1416e+03
%           AIC: -2.0355e+03
%           regressors: [nxK]
%           predict_prob: [nxK]: Pr(zt=k|y1...y_{t-1})
%           predicted: [nx1]
%           filter_prob: [nxK]: Pr(zt=k|y1...y_t)
%           filtered: [nx1]
%           smoothed_regressors: [nxK]
%           smoothed: [nx1]
%
%
%Faicel Chamroukhi, sept 2008 
%
%% Please cite the following papers for this code:
%
% 
% @article{Chamroukhi-FDA-2018,
% 	Journal = {Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery},
% 	Author = {Faicel Chamroukhi and Hien D. Nguyen},
% 	Note = {DOI: 10.1002/widm.1298.},
% 	Volume = {},
% 	Title = {Model-Based Clustering and Classification of Functional Data},
% 	Year = {2019},
% 	Month = {to appear},
% 	url =  {https://chamroukhi.com/papers/MBCC-FDA.pdf}
% 	}
% 
% @InProceedings{Chamroukhi-IJCNN-2011,
%   author = {F. Chamroukhi and A. Sam\'e  and P. Aknin and G. Govaert},
%   title = {Model-based clustering with Hidden Markov Model regression for time series with regime changes},
%   Booktitle = {Proceedings of the International Joint Conference on Neural Networks (IJCNN), IEEE},
%   Pages = {2814--2821},
%   Adress = {San Jose, California, USA},
%   year = {2011},
%   month = {Jul-Aug},
%   url = {https://chamroukhi.com/papers/Chamroukhi-ijcnn-2011.pdf}
% }
% 
% @INPROCEEDINGS{Chamroukhi-IJCNN-2009,
%   AUTHOR =       {Chamroukhi, F. and Sam\'e,  A. and Govaert, G. and Aknin, P.},
%   TITLE =        {A regression model with a hidden logistic process for feature extraction from time series},
%   BOOKTITLE =    {International Joint Conference on Neural Networks (IJCNN)},
%   YEAR =         {2009},
%   month = {June},
%   pages = {489--496},
%   Address = {Atlanta, GA},
%  url = {https://chamroukhi.com/papers/chamroukhi_ijcnn2009.pdf},
%  slides = {./conf-presentations/presentation_IJCNN2009}
% }
% 
% @article{chamroukhi_et_al_NN2009,
% 	Address = {Oxford, UK, UK},
% 	Author = {Chamroukhi, F. and Sam\'{e}, A. and Govaert, G. and Aknin, P.},
% 	Date-Added = {2014-10-22 20:08:41 +0000},
% 	Date-Modified = {2014-10-22 20:08:41 +0000},
% 	Journal = {Neural Networks},
% 	Number = {5-6},
% 	Pages = {593--602},
% 	Publisher = {Elsevier Science Ltd.},
% 	Title = {Time series modeling by a regression approach based on a latent process},
% 	Volume = {22},
% 	Year = {2009},
% 	url  = {https://chamroukhi.users.lmno.cnrs.fr/papers/Chamroukhi_Neural_Networks_2009.pdf}
% 	}
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off

if nargin<9 verbose =0; end
if nargin<8 threshold = 1e-6; end
if nargin<7 max_iter_EM = 1500;end
if nargin<6 total_EM_tries = 1;end
if nargin<5 total_EM_tries = 1;type_variance='hetereskedastic';end
switch type_variance
    case 'homoskedastic'
        homoskedastic =1;
    case 'hetereskedastic'
        homoskedastic=0;
    otherwise
        error('The type of the model variance should be : ''homoskedastic'' ou ''hetereskedastic''');
end

if size(y,2)~=1, y=y'; end %

m = length(y);%length(y)

X = designmatrix(x,p);%design matrix
P = size(X,2);% here P is p+1
I = eye(P);% define an identity matrix, in case of a Bayesian regularization for regression

%
best_loglik = -inf;
nb_good_try=0;
total_nb_try=0;

while (nb_good_try < total_EM_tries)
    if total_EM_tries>1,fprintf(1, 'EM try n∞  %d \n ',nb_good_try+1); end
    total_nb_try=total_nb_try+1;
    
    time = cputime;
    
    %% EM Initializaiton step
    %% Initialization of the Markov chain params, the regression coeffs, and the variance(s)
    hmmr =  init_hmmr(X, y, K, type_variance, nb_good_try+1);
    
    % calculare the initial post probs  (tau_tk) and joint post probs (xi_ikl)
    
    %f_tk = hmmr.stats.f_tk; % observation component densities: f(yt|zt=k)
    prior = hmmr.prior;
    trans_mat = hmmr.trans_mat;
    Mask = hmmr.stats.Mask;
    
    betak = hmmr.reg_param.betak;
    if homoskedastic
        sigma2 = hmmr.reg_param.sigma2 ;
    else
        sigma2k = hmmr.reg_param.sigma2k;
    end
    
    %
    iter = 0;
    prev_loglik = -inf;
    converged = 0;
    top = 0;
    
    %
    log_f_tk = zeros(m,K);
    muk = zeros(m, K);
    %
    %% EM
    while ((iter <= max_iter_EM) && ~converged)
        
        %% E step : calculate tge tau_tk (p(Zt=k|y1...ym;theta)) and xi t_kl (and the log-likelihood) by
        %  forwards backwards (computes the alpha_tk et beta_tk)
        
        % observation likelihoods
        for k=1:K
            mk = X*betak(:,k);
            muk(:,k) = mk; % the regressors means
            if homoskedastic; sk = sigma2 ;  else; sk = sigma2k(k);end
            z=((y - mk).^2)/sk;
            log_f_tk(:,k) =  -0.5*ones(m,1).*(log(2*pi)+log(sk)) - 0.5*z;%log(gaussienne)
        end
        
        log_f_tk  = min(log_f_tk,log(realmax));
        log_f_tk = max(log_f_tk ,log(realmin));
        f_tk = exp(log_f_tk);
        
        %fprintf(1, 'forwards-backwards ');
        [tau_tk, xi_tkl, alpha_tk, beta_tk, loglik] = forwards_backwards(prior, trans_mat , f_tk );
        
        %% M step
        %  updates of the Markov chain parameters
        % initial states prob: P(Z_1 = k)
        prior = normalise(tau_tk(1,:)');
        % transition matrix: P(Zt=i|Zt-1=j) (A_{k\ell})
        trans_mat = mk_stochastic(squeeze(sum(xi_tkl,1)));
        % for segmental HMMR: p(z_t = k| z_{t-1} = \ell) = zero if k<\ell (no back) of if k >= \ell+2 (no jumps)
        trans_mat = mk_stochastic(Mask.*trans_mat);
        
        %%  update of the regressors (reg coefficients betak and the variance(s) sigma2k)
        
        s = 0;% if homoskedastic
        for k=1:K
            wieghts = tau_tk(:,k);
            nk = sum(wieghts);% expected cardinal nbr of state k
            
            Xk = X.*(sqrt(wieghts)*ones(1,P));%[n*(p+1)]
            yk=y.*(sqrt(wieghts));% dimension :[(nx1).*(nx1)] = [nx1]
            % reg coefficients
            lambda = 1e-9;% if a bayesian prior on the beta's
            bk = inv(Xk'*Xk + lambda*I)*Xk'*yk;
            betak(:,k) = bk;
            % variance(s)
            z = sqrt(wieghts).*(y-X*bk);
            sk = z'*z;
            if homoskedastic, s = (s+sk); sigma2 = s/m;else; sigma2k(k) = sk/nk;end
        end
        
        %% En of an EM iteration
        iter =  iter + 1;
        
        % test of convergence
        loglik = loglik + log(lambda);
        
        if verbose, fprintf(1, 'HMM_regression | EM   : Iteration : %d   Log-likelihood : %f \n',  iter, loglik); end
        
        if prev_loglik-loglik > 1e-4
            top = top+1;
            if (top==10)
                %fprintf(1, '!!!!! The loglikelihood is decreasing from %6.4f to %6.4f!\n', prev_loglik, loglik);
                break;
            end
        end
        converged = abs(loglik - prev_loglik)/abs(prev_loglik) < threshold;
        stored_loglik(iter) = loglik;
        prev_loglik = loglik;
        
    end% end of an EM run
    
    cputime_total(nb_good_try+1) = cputime-time;
    
    hmmr.prior = prior;
    hmmr.trans_mat = trans_mat;
    hmmr.reg_param.betak = betak;
    if homoskedastic
        hmmr.reg_param.sigma2 = sigma2;
    else
        hmmr.reg_param.sigma2k = sigma2k;
    end
    
    % Estimated parameter vector (Pi,A,\theta)
    if homoskedastic
        parameter_vector=[prior(:); trans_mat(Mask~=0); betak(:);sigma2];
        nu = K-1 + K*(K-1) + K*(p+1) + 1;%length(parameter_vector);%
    else
        parameter_vector=[prior(:); trans_mat(Mask~=0);betak(:); sigma2k(:)];
        nu = K-1 + K*(K-1) + K*(p+1) + K;%length(parameter_vector);%
    end
    hmmr.stats.nu = nu;
    hmmr.stats.parameter_vector= parameter_vector;
    
    hmmr.stats.tau_tk = tau_tk; % posterior (smoothing) probs
    hmmr.stats.alpha_tk = alpha_tk;%forward probs
    hmmr.stats.beta_tk = beta_tk;%backward probs
    hmmr.stats.xi_ikl = xi_tkl;% joint posterior (smoothing) probs
    
    hmmr.stats.f_tk = f_tk;% obs likelihoods
    hmmr.stats.log_f_tk = log_f_tk;% log obs likelihoods
    
    hmmr.stats.loglik= loglik;
    hmmr.stats.stored_loglik= stored_loglik;
    %
    hmmr.stats.X = X;%design matrix
    %
    if total_EM_tries>1,   fprintf(1,'loglik_max = %f \n',loglik); end
    %
    %
    if ~isempty(hmmr.reg_param.betak)
        nb_good_try=nb_good_try+1;
        total_nb_try=0;
        if loglik > best_loglik
            best_hmmr = hmmr;
            best_loglik = loglik;
        end
    end
    %
    if total_nb_try > 500
        fprintf('can''t obtain the requested number of classes \n');
        hmmr=[];
        return;
    end
    
end%End of the EM runs

hmmr = best_hmmr;

%
if total_EM_tries>1,    fprintf(1,'best_loglik:  %f\n',hmmr.stats.loglik);end
%
%
hmmr.stats.cputime = mean(cputime_total);
hmmr.stats.cputime_total = cputime_total;

%% Smoothing state sequences : argmax(smoothing probs), and corresponding binary allocations partition
[hmmr.stats.klas, hmmr.stats.Zik ] =  MAP(hmmr.stats.tau_tk);
% %  compute the sequence with viterbi
%[path, ~] = viterbi_path(hmmr.prior, hmmr.trans_mat, hmmr.stats.fik');
%hmmr.stats.viterbi_path = path;
%hmmr.stats.klas = path;
%%%%%%%%%%%%%%%%%%%

% %% determination des temps de changements (les fonti√®tres entres les
% %% classes)
% nk=sum(hmmr.stats.Zik,1);
% for k = 1:K
%     tk(k) = sum(nk(1:k));
% end
% hmmr.stats.tk = [1 tk];

%% sate sequence prob p(z_1,...,z_n;\pi,A)
state_probs = hmm_process(hmmr.prior, hmmr.trans_mat, m);
hmmr.stats.state_probs = state_probs;


%%% BIC, AIC, ICL
hmmr.stats.BIC = hmmr.stats.loglik - (hmmr.stats.nu*log(m)/2);
hmmr.stats.AIC = hmmr.stats.loglik - hmmr.stats.nu;
% % CL(theta) : Completed-data loglikelihood
% sum_t_log_Pz_ftk = sum(hmmr.stats.Zik.*log(state_probs.*hmmr.stats.f_tk), 2);
% comp_loglik = sum(sum_t_log_Pz_ftk(K:end));
% hmmr.stats.comp_loglik = comp_loglik;
% hmmr.stats.ICL = comp_loglik - (nu*log(m)/2);


%% predicted, filtered, and smoothed time series
hmmr.stats.regressors = X*hmmr.reg_param.betak;
% prediction probs   = Pr(z_t|y_1,...,y_{t-1})
predict_prob = zeros(m,K);
predict_prob(1,:) = hmmr.prior;%t=1 p (z_1)
predict_prob(2:end,:) = (hmmr.stats.alpha_tk(1:end-1,:)*hmmr.trans_mat)./(sum(hmmr.stats.alpha_tk(1:end-1,:),2)*ones(1,K));%t =2,...,n
hmmr.stats.predict_prob = predict_prob;
% predicted observations
hmmr.stats.predicted = sum(predict_prob.*hmmr.stats.regressors,2);%pond par les probas de prediction

% filtering probs  = Pr(z_t|y_1,...,y_t)
filter_prob = hmmr.stats.alpha_tk./(sum(hmmr.stats.alpha_tk,2)*ones(1,K));%normalize(alpha_tk,2);
hmmr.stats.filter_prob = filter_prob;
% filetered observations
hmmr.stats.filtered = sum(filter_prob.*hmmr.stats.regressors, 2);%pond par les probas de filtrage

%%% smoothed observations
hmmr.stats.smoothed_regressors = (hmmr.stats.tau_tk).*(hmmr.stats.regressors);
hmmr.stats.smoothed = sum(hmmr.stats.smoothed_regressors, 2);












