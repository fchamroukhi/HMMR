function [M, c] = normalise(M)
% NORMALISE Make the entries of a (multidimensional) array sum to 1
% [M, c] = normalise(M)
%
% This function uses the British spelling to avoid any confusion with
% 'normalize_pot', which is a method for potentials.

c = sum(M(:));
% Set any zeros to one before dividing
d = c + (c==0);
M = M / d;
