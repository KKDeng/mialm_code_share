function [U, V, stats] = rmc(problem, params, init)
%% RMC Robust Matrix Completion
%
% [U, V, stats] = rmc(problem, params, init) run the RMC algorithm
% - problem is a structure coming from the buildProblemL1 function
%   ===> the parameter lambda (regularization) is required and should be provided
%   in addition : problem.lambda = 0 ;
% - params is a structure with the following fields (all required) 
%       params.manopt.maxiter = 40         inner-iteration for CG
%       params.manopt.verbosity = 2        verbosity of CG
%       params.manopt.minstepsize = 0      minimum step size of CG
%       params.manopt.tolgradnorm = 1e-8   gradient tolerance of CG
%       params.huber.epsilon = 1           the initial value for delta
%       params.huber.theta = 0.05          the theta parameter
%       params.huber.tol = 1e-8            the tolerance for RMC
%       params.huber.itmax = 7             the maximum number of iterations
%       params.huber.verbose = 1           verbosity of CG
% - init is either true, false or a structure. It can be ommited
%       init = (nothing) : initialization using SVD of P(M)
%       init = true : initialization random
%       init = false : initialization using SVD of P(M)
%       init = struct (init.U, init.V, init.S, X = U S V') : warm-up init 
%       (U, V orthonormal, S diagonal full rank) using X. The rank of X,
%       i.e. the size of S, should be at most problem.r
% It returns the following :
% - U, V such that U*V is the recovered low-rank matrix
% - stats contains multiple fields with informations on the algorithm
%   convergence. It is an array of structure ;
%       [stats.Time]     the time of each point on the MAE/RMSE/RMSEtest curve
%       [stats.MAE]      the MAE (mean absolute error) with respect to the
%                        original matrix on the mask omega
%       [stats.nOuts]    the number of outliers (tolerance 1e-4)
%       [stats.RMSE]     the RMSE (root mean square error) with respect to
%                        the original matrix on all the entries
%       [stats.RMSEtest] the RMSEtest (if Atest, Itest and Jtest provided 
%                        in problem, in addition to the original fields)
%       [stats.Its]      the iteration at which epsilon (delta) changes
%       [stats.Obj]      the objective at each iteration
%       [stats.Epsilon]  the value of epsilon (delta)
%
% This function requires the manopt toolbox (www.manopt.org) and was tested
% using Manopt 1.0.7
% It also requires the content of the common folder.
%
% Original code by Leopold Cambier, 2015

m = problem.m ;
n = problem.n ;
r = problem.r ;

% Manifold : pick a geometry
problemManopt.M = fixedrankembeddedfactory(m, n, r); 
problem.M = problemManopt.M ;

% Params Manopt solver
opts.maxiter = params.manopt.maxiter ;
opts.verbosity = params.manopt.verbosity ;
opts.minstepsize = params.manopt.minstepsize ;
opts.tolgradnorm = params.manopt.tolgradnorm ;
opts.statsfun = @(optproblem, X, stats, store) statsHuber(optproblem, X, stats, store, problem) ;

% Initial point
if nargin == 2
    init = false ;
end

% Choose the initial point type
initialPointTime = tic ;
if ~isstruct(init) && isscalar(init)
    if init % Full random
        U = rand(m, r) ;
        [U,~] = qr(U,0) ;    
        V = rand(n, r) ;
        [V,~] = qr(V,0) ;    
        if params.huber.verbose >= 1
            fprintf('Using random initialization\n') ;
        end
        S = diag(mean(abs(problem.A)) * rand(r, 1) .* sign(rand(r, 1) - 0.5)) ;
    else % SVD of P(M)
        [U, S, V] = svds(sparse(problem.I, problem.J, problem.A, m, n), r) ;
        if params.huber.verbose >= 1
            fprintf('Using SVD initialization\n') ;
        end
    end
elseif isstruct(init) % Hot warm-up
    if isfield(init, 'U') && isfield(init, 'S') && isfield(init, 'V')
        if params.huber.verbose >= 1
            fprintf('Using warm-up initialization with a matrix of rank %d\n',size(init.U, 2)) ;
        end
        Uinit = init.U ; % Orthonormal
        Vinit = init.V ; % Orthonormal
        Sinit = init.S ; % Diagonal, full rank
        rinit = size(Uinit, 2) ;
        r = problem.r ;
        if size(Vinit, 2) ~= rinit || any(size(Sinit) ~= [rinit rinit]) || size(Uinit, 1) ~= problem.m || size(Vinit, 1) ~= problem.n
            error('Wrong sizes for U, V and S') ;
        end
        if r < rinit
            error('r < rinit !') ;
        end
        % Create random vectors for U and V
        Up = randn(m, r - rinit) ;
        Up = Up - Uinit*(Uinit' * Up) ; % Project it
        [Up, ~] = qr(Up, 0) ; % Orthogonalise it
        Vp = randn(n, r - rinit) ;         
        Vp = Vp - Vinit*(Vinit' * Vp) ; % Project it
        [Vp, ~] = qr(Vp, 0) ; % Orthogonalise it
        U = [Uinit Up] ;
        V = [Vinit Vp] ;
        S = diag([diag(Sinit) ; ones(r - rinit, 1)*mean(diag(Sinit))]) ;
        % Check orthogonality and full-rankness
        norm(U'*U - eye(r))
        norm(V'*V - eye(r))
    else
        error('Wrong init argument : should be a structure with fields U, S and V') ;
    end
else
    error('Wrong init argument : should be true, false or a structure') ;
end

X.U = U;
X.S = S;
X.V = V;

% Initial stats
stats(1).Time = toc(initialPointTime) ;
UV = spmaskmult(U*S, V', problem.Iu, problem.Ju) ;
stats(1).MAE = mean(abs(problem.Atrue - UV)) ;
diff = abs(problem.A - UV) ;
stats(1).nOuts = countInexactRecoveries(diff, problem.Atrue) ;
stats(1).RMSE = sqrt(sqfrobnormfactors(U*S,V',problem.Utrue,problem.Vtrue)/(m*n)) ;
stats(1).RMSEtest = nan ;
stats(1).Its = 1 ;

% Solve
iteration = 1 ;
obj = inf ;
convCriterion = inf ;
epsilon = params.huber.epsilon ;
theta = params.huber.theta ;

fprintf('Huber iteration running ...\n') ;
while iteration <= params.huber.itmax && convCriterion > params.huber.tol
        
    % Prepare problem
    problem.delta = epsilon ;    
    problemManopt.cost = @(X, store) costHuber(X, store, problem) ;
    problemManopt.grad = @(X, store) gradHuber(X, store, problem) ;    
    
    % Solve      
    [Xnew, ~, statsSolver] = conjugategradient(problemManopt, X, opts) ;          
            
    % Diagnostics
    stats(iteration+1).MAE  = [statsSolver.MAE] ; 
    stats(iteration+1).Obj  = [statsSolver.Obj] ; 
    stats(iteration+1).nOuts = [statsSolver.nOut] ; 
    stats(iteration+1).RMSE = [statsSolver.RMSE] ; 
    stats(iteration+1).Its = length(statsSolver) + stats(iteration).Its ;
    oldTime = stats(iteration).Time ;
    stats(iteration+1).Time = [statsSolver.time] + oldTime(end) ;            
    stats(iteration+1).Epsilon = epsilon ;
    if isfield(statsSolver, 'RMSEtest')
        stats(iteration+1).RMSEtest = [statsSolver.RMSEtest] ;
    else
        stats(iteration+1).RMSEtest = nan ;
    end
    
    % Print and compute new
    objnew = statsSolver(end).Obj ;
    maenew = statsSolver(end).MAE ;

    % Print stuff
    if params.huber.verbose > 0
        fprintf('Huber Status (it. %d) : obj = %e, epsilon = %e, MAE = %e\n', iteration, objnew, epsilon, maenew) ;
    end
            
    % Update
    epsilon = epsilon * theta ;
    X = Xnew ;
    convCriterion = abs(obj - objnew) ;
    obj = objnew ; 
    iteration = iteration + 1 ;
end

U = X.U*X.S ;
V = X.V' ;

if params.huber.verbose > 0
    fprintf('Huber Status : terminated.\n') ;
end

end

function stats = statsHuber(optproblem, X, stats, store, prob)
    
    A = prob.A ; 
    Atrue = prob.Atrue ;

    prob.delta = 1 ; % Just for the store
    store = updateStore(X, store, prob) ;
    
    Xdiffabs = abs(store.XmatIJdiff) ;
    if ~ isempty(Atrue)
        Xdiffabstrue = abs(store.XmatIJ - Atrue) ;    
        stats.MAE = mean(Xdiffabstrue) ;
        stats.nOut = sum(Xdiffabs >= 1e-4 * abs(Atrue)) ;
    else
        stats.MAE = nan ;
        stats.nOut = sum(Xdiffabs >= 1e-4 * abs(A)) ;
    end    
    stats.Obj = mean(Xdiffabs) ;
    
    stats.RMSE = sqrt(sqfrobnormfactors(prob.Utrue, prob.Vtrue, store.U, store.V) / prob.m / prob.n) ; 
 
    if isfield(prob, 'Atest')
        % prob.Atest, prob.Itest, prob.Jtest need to be in prob
        if any(size(prob.Atest) ~= size(prob.Itest)) || any(size(prob.Atest) ~= size(prob.Jtest))
            error('Wrong size in Atest, Itest or Jtest') ;
        end
        diffTest = spmaskmult(store.U, store.V, prob.Itest, prob.Jtest) - prob.Atest ;
        RMSEtest = sqrt(sum(diffTest(:).^2)/numel(prob.Atest)) ;
        U = store.U ;
        V = store.V ;
        Itest = prob.Itest ;
        Jtest = prob.Jtest ;
        Atest = prob.Atest ;
        save(sprintf('RMSEtest_lambda_%f_r_%d_RMSEtest_%f.mat',prob.lambda, prob.r, RMSEtest), 'U', 'V', 'Itest', 'Jtest', 'Atest', 'RMSEtest') ;
        stats.RMSEtest = RMSEtest ;        
    end
    
end

% The cost function in manopt format
function [f, store] = costHuber(X, store, prob)
if ~isfield(store, 'val')         
    store = updateStore(X, store, prob) ;    
    lambda = prob.lambda ;
    k = prob.k ;
    store.val =  sum(store.XmatIJsqrt - lambda .* store.XmatIJ.^2) / k ...
               + lambda * norm(X.S,'fro')^2 / k ;
end
f = store.val ;
end

% The gradient in manopt format
% This is already the Riemannian gradient
function [grad, store] = gradHuber(X, store, prob)
if ~isfield(store, 'grad')       
    m = prob.m ;
    n = prob.n ;
    k = prob.k ;
    I = prob.I ;
    J = prob.J ;         
    lambda = prob.lambda ;
    store = updateStore(X, store, prob) ;        
    Z = store.XmatIJdiff ./ store.XmatIJsqrt / k ...
        - 2 * lambda * store.XmatIJ / k ;
    Z = sparse(I, J, Z, m, n) ;    
    U = X.U ;
    S = X.S ;
    V = X.V ;   
    M = U'*(Z*V) + 2*lambda*S/k ;
    UtZV = U'*(Z*V) ;
    Up = Z*V - U*UtZV ;
    Vp = Z'*U - V*UtZV' ;   
    store.grad.M = M ;
    store.grad.Up = Up ;
    store.grad.Vp = Vp ;   
end
grad = store.grad ;   
end

function store = updateStore(X, store, prob)
Iu = prob.Iu ;
Ju = prob.Ju ;
AIJ = prob.A ;
delta = prob.delta ;    
if ~ isfield(store, 'XmatIJ')    
    US = X.U*X.S ;
    V = X.V' ;
        
    store.U = US ;
    store.V = V ;
    store.XmatIJ = spmaskmult(US, V, Iu, Ju) ;                   
end
if ~ isfield(store, 'XmatIJdiff') 
    store.XmatIJdiff = store.XmatIJ - AIJ ;
end
if ~ isfield(store, 'XmatIJsqrt') 
    store.XmatIJsqrt = sqrt(delta.^2 + store.XmatIJdiff.^2) ; 
end
end


