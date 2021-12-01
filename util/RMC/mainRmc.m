%% Run an RMC example
% You need the following folder in path :
% - common (including the subfolder)
% - manopt (1.0.7 works fine)
clc ; clear all ; close all ;

% Parameters
% CG parameters
params.manopt.maxiter = 40 ;
params.manopt.verbosity = 2 ;
params.manopt.minstepsize = 0 ;
params.manopt.tolgradnorm = 1e-8 ;
% Outer loop parameters
params.huber.epsilon = 1 ;  % Good for synthetic experiments. For other problems, use the mean absolute value of the matrix M
params.huber.theta = 0.05 ; % Good for synthetic experiments. Can be increased (0.1, 1/2, etc.) for real datasets.
params.huber.tol = 1e-8 ; 
params.huber.itmax = 7 ;    % The maximum number of iteration
params.huber.verbose = 1 ;

% Problem setup : a 1000 x 1000 rank-10 matrix completion problem
m = 1000 ;
n = 1000 ;
r = 10 ;
[Utrue, Vtrue, Xtrue, I, J, k] = generateSyntheticData(m, n, r, 5*r*(m+n-r)) ; % Oversampling of 5
% Add 5% outliers (additive, sign and uniform are used to change the
% outliers distribution : you can keep these values as it)
[Xout, nOut] = addOutliers(I, J, Xtrue, m, n, 0.05, 1, 1, 'additive', 'sign', 'uniform') ;

% Build the problem
problemHuber = buildProblemL1(I, J, Xout, m, n, r, Utrue, Vtrue) ;
% Choose a regularization parameter (0 is perfectly fine for synthetic experiments)
problemHuber.lambda = 0 ;

% Run RMC
[U, V, stats] = rmc(problemHuber, params) ;

% Plot the evolution of the RMSE (i.e. the error) with respect to the
% original matrix)
semilogy([stats.Time],[stats.RMSE]) ;