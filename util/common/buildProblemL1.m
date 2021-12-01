function problem = buildProblemL1(I, J, A, m, n, r, in1, in2)
% buildProblemL1(I, J, A, m, n, r, Utrue, Vtrue)
% or
% buildProblemL1(I, J, A, m, n, r, Atrue)
% where A are the observed entries on mask (I, J), the matrix is of size m
% x n with a rank r.
% Atrue is the real matrix on omega, while Utrue*Vtrue is his factorization
% problem is a structure to be used by rmc, irls and almc.

if any(size(I) ~= size(A)) || any(size(I) ~= size(J))
    error('Wrong sizes') ;
end

if nargin == 6
    Utrue = ones(m, r) ;
    Vtrue = ones(r, n) ;
    Atrue = ones(size(I)) ;
    warning('Using random Utrue and Vtrue. MAE and RMSE will be meaningless.') ;   
elseif nargin == 7
    Utrue = ones(m, r) ;
    Vtrue = ones(r, n) ;
    if any(size(in1) ~= size(I)) ;
        error('Wrong size') ;
    end
    Atrue = in1 ;
    warning('Using random Utrue and Vtrue. RMSE will be meaningless.') ;   
elseif nargin == 8
    Utrue = in1 ;
    Vtrue = in2 ;
    if any(size(in1) ~= [m r]) || any(size(in2) ~= [r n]) ;
        error('Wrong size') ;
    end
    Atrue = spmaskmult(Utrue, Vtrue, uint32(I), uint32(J)) ;
end

problem.Iu = uint32(I) ;
problem.Ju = uint32(J) ;
problem.I = double(I) ;
problem.J = double(J) ;
problem.A = A ;
problem.m = m ;
problem.n = n ;
problem.r = r ;
problem.k = numel(A) ;

problem.Utrue = Utrue ;
problem.Vtrue = Vtrue ;
problem.Atrue = Atrue ;





end