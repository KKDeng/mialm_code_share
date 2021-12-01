function [Utrue, Vtrue, Xtrue, Iu, Ju, k] = generateSyntheticDataUniform(m, n, r, k)
% Generate synthetics data for experiments on a m x n matrix of rank r with
% an k observed entries
% Entries of Xtrue are Gaussian variable with mean 0 and variance 1
% The outputs are the matrix factorization (Utrue*Vtrue is the matrix),
% Xtrue is Utrue*Vtrue on the mask (Iu, Ju).
% In case if k is not exact, the new value is returned.

fprintf('Generating synthetic data with uniform in U and V...') ;

if nargin <= 3
    k = 4*r*(m+n-r);
end

if k > m*n 
    error('k is too large ...') ;
end


Utrue = sqrt(12)*(rand(m, r)-0.5) ; % zero-mean, 1/r-variance
Vtrue = sqrt(12)*(rand(r, n)-0.5) ; % zero-mean, 1/r-variance
[Iu, Ju, k] = randmask(m, n, k);
Xtrue = spmaskmult(Utrue, Vtrue, Iu, Ju);

if k > m*n 
    error('k is too large ...') ;
end

perm = randperm(k);
Iu = Iu(perm);
Ju = Ju(perm);
Xtrue = Xtrue(perm);

fprintf(' %dx%d with %d/%d (%4.2f pc) known elements.\n',m, n, k, m*n, k/m/n*100) ;
end