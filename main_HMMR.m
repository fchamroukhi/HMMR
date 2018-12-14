%
% Hidden Markov Model Regression (HMMR) for segmentation of time series
% with regime changes. The model assumes that the time series is
% governed by a sequence of hidden discrete regimes/states, where each
% regime/state has Gaussian regressors as observations.
% The model parameters are estimated by MLE via the EM algorithm
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
% type_variance = 'homoskedastic';
type_variance = 'hetereskedastic';
nbr_EM_tries = 2;
max_iter_EM = 1500;
threshold = 1e-6;
verbose = 1;
type_algo = 'EM';
% type_algo = 'CEM';
% type_algo = 'SEM';

%% toy time series with regime changes
y =[randn(100,1); 7+randn(120,1);4+randn(200,1); -2+randn(100,1); 3.5+randn(150,1);]';
n = length(y);
x = linspace(0,1,n);

%load simulated_time_series;

HMMR = learn_hmmr(x, y, K, p, type_variance, nbr_EM_tries, max_iter_EM, threshold, verbose);

%     %if model selection
%     current_BIC = -inf;
%     for K=1:8
%         for p=0:4
%             HMMR_K = learn_hmmr(x, y, K, p, type_variance, nbr_EM_tries, max_iter_EM, threshold, verbose)
%
%             if HMMR_K.stats.BIC>current_BIC
%                 HMMR=HMMR_K;
%                 current_BIC = HMMR_K.stats.BIC;
%             end
%                 bic(K,p+1) = HMMR_K.stats.BIC;
%         end
%     end
show_HMMR_results(x,y, HMMR)

%load real_time_series_1
load real_time_series_2
HMMR = learn_hmmr(x, y, K, p, type_variance, nbr_EM_tries, max_iter_EM, threshold, verbose);
yaxislim= [240 600];
show_HMMR_results(x,y, HMMR, yaxislim)




