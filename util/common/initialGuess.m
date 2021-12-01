function U0 = initialGuess(I, J, X, m, n, r)     
% The initial guess U0 for the Grassmanian algorithm
[U0, ~, ~] = svds(sparse(I, J, X, m, n, numel(X)), r) ;
end