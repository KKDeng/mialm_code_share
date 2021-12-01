function [U, V, stats] = easySolveL2(I, J, X, m, n, r, lambda)
% Runs RTRMC (see in-code for parameters)

if nargin == 6
    warning('No lambda provided ; lambda set to 0') ;
    lambda = 0 ;
end

opts.method = 'rtr';     
opts.order = 2;          
opts.precon = true;      
opts.maxiter = 200;      
opts.maxinner = 300;      
opts.tolgradnorm = 1e-8; 
opts.verbosity = 2;      
opts.computeRMSE = false; 
opts.computeMAE = false;

problemRtrmc = buildproblem(I, J, X, ones(size(X)), m, n, r, lambda) ;  
problemRtrmc.A = [] ; % the computeRMSE = false doesn't work :/
problemRtrmc.B = [] ; 

U = initialguess(problemRtrmc);

[U, V, stats] = rtrmc(problemRtrmc, opts, U) ;


end