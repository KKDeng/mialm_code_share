function [U, V] = randomInit(m, n, r)
% A random initialization of a m x n matrix of rank r
% With unit-variance, zero-mean gaussian variables
U = randn(m, r)/r.^.25 ;
V = randn(r, n)/r.^.25 ;
end