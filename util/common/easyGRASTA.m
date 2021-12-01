function [U, V, stats] = easyGRASTA(problem, params)
% Runs grasta (see in-code for parameters)

if problem.m > problem.n
    warning('Grasta is optimized for m <= n ; please transpose your problem.') ;
end

fprintf('Running GRASTA ...\n') ;

maxCycles                   = 50 ;     % the max cycles of robust mc
OPTIONS.QUIET               = 1 ;      % suppress the debug information

OPTIONS.MAX_LEVEL           = 20 ;     % For multi-level step-size,
OPTIONS.MAX_MU              = 15 ;     % For multi-level step-size
OPTIONS.MIN_MU              = 1  ;     % For multi-level step-size

OPTIONS.DIM_M               = problem.m ;  % your data's ambient dimension
OPTIONS.RANK                = problem.r ;  % give your estimated rank

OPTIONS.ITER_MIN            = 20 ;      % the min iteration allowed for ADMM at the beginning
OPTIONS.ITER_MAX            = 20 ;      % the max iteration allowed for ADMM
OPTIONS.rho                 = 2  ;      % ADMM penalty parameter for acclerated convergence
OPTIONS.TOL                 = 1e-8 ;   % ADMM convergence tolerance

OPTIONS.USE_MEX             = 1 ;     % If you do not have the mex-version of Alg 2
                                     % please set Use_mex = 0.
warning('USE_MEX set to 1') ;                                     
                                     
CONVERGE_LEVLE              = 20 ;    % If status.level >= CONVERGE_LEVLE, robust mc converges

[Usg, Vsg, Osg, stats] = grasta_mc(problem.I, problem.J, problem.A, ...
                                   problem.m, problem.n, maxCycles, ...
                                   CONVERGE_LEVLE, OPTIONS, problem, params) ;
                        
U = Usg ;
V = Vsg' ;

fprintf('GRASTA terminated.\n') ;

end