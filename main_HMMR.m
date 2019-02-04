% User-freindly and flexible model anf algorithm for time series segmentation with a Regression
% model with a Hidden Markov Model Regression (HMMR).
%
%
% Hidden Markov Model Regression (HMMR) for segmentation of time series
% with regime changes. The model assumes that the time series is
% governed by a sequence of hidden discrete regimes/states, where each
% regime/state has Gaussian regressors as observations.
% The model parameters are estimated by MLE via the EM algorithm
%
% Faicel Chamroukhi
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
%  url = {https://chamroukhi.com/papers/chamroukhi_ijcnn2009.pdf}
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
% 	url  = {https://chamroukhi.com/papers/Chamroukhi_Neural_Networks_2009.pdf}
% 	}
% 
% 
%
%
% Faicel Chamroukhi Septembre 2008.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc

% model specification
K = 5; % nomber of regimes (states)
p = 3; % dimension of beta' (order of the polynomial regressors)

% options
%type_variance = 'homoskedastic';
type_variance = 'hetereskedastic';
nbr_EM_tries = 1;
max_iter_EM = 1500;
threshold = 1e-6;
verbose = 1;
type_algo = 'EM';
% type_algo = 'CEM';
% type_algo = 'SEM';

%% toy time series with regime changes
% y =[randn(100,1); 7+randn(120,1);4+randn(200,1); -2+randn(100,1); 3.5+randn(150,1);]';
% n = length(y);
% x = linspace(0,1,n);

load simulated_time_series;
%load real_time_series_1

HMMR = learn_hmmr(x, y, K, p, type_variance, nbr_EM_tries, max_iter_EM, threshold, verbose);

%     %if model selection
%     current_BIC = -inf;
%     for K=1:8
%         for p=0:4
%             HMMR_Kp = learn_hmmr(x, y, K, p, type_variance, nbr_EM_tries, max_iter_EM, threshold, verbose)
% 
%             if HMMR_Kp.stats.BIC>current_BIC
%                 HMMR=HMMR_Kp;
%                 current_BIC = HMMR_Kp.stats.BIC;
%             end
%                 bic(K,p+1) = HMMR_Kp.stats.BIC;
%         end
%     end
show_HMMR_results(x,y, HMMR)


load real_time_series_1
%load real_time_series_2

HMMR = learn_hmmr(x, y, K, p, type_variance, nbr_EM_tries, max_iter_EM, threshold, verbose);
yaxislim= [240 600];
show_HMMR_results(x,y, HMMR, yaxislim)


% sample an HMMR
%[y, states, Z, mean_function] = sample_hmmr(x, HMMR.prior, HMMR.trans_mat, HMMR.reg_param.betak,HMMR.reg_param.sigma2k);

